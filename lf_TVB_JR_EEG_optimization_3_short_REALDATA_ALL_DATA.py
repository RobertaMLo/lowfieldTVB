import os

import numpy as np
from tvb.simulator.lab import *
import pandas as pd


from utils_optim import *
from utils_myoptimizer import *
from utils_mypsd import *
import argparse
from abc import ABC


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- CLASSE DI OTTIMIZZAZIONE ------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

class TVBOptimizationProblem(Problem, ABC):
    """
    Problem with 2 variables: [global_coupling (G), lp]
    Objectives:
      F[:,0] = errore PSD ratio
      F[:,1] = errore media segnale
    """

    def __init__(self,
                 exp_eeg,
                 tvb_cfg,
                 con,
                 heunint,
                 my_mon,
                 regions_name,
                 fsamp,
                 xl=None,
                 xu=None):
        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)

        self.sim_len = tvb_cfg["sim_len"]
        self.con = con
        self.exp_eeg = exp_eeg
        self.heunint = heunint
        self.mon_eeg = my_mon

        self.bands, _ = get_freqs_bands()
        self.threshold_ratio = 0.0
        self.regions_name = [f"ch{i}" for i in range(len(regions_name))]

        self.fs = fsamp
        self.window_length = fsamp * 2
        self.overlap = 0
        self.fc = 45
        self.order = 4

    def simulate_TVB(self, global_coupling, lp):
        tau_e, tau_i = update_tau_from_lp(lp)
        a = 1.0 / tau_e
        b = 1.0 / tau_i

        mod = models.JansenRit(mu=np.array(0.9), v0=np.array(6.0),
                               a=np.array([a]), b=np.array([b]))
        G = np.array(global_coupling)

        sim = simulator.Simulator(
            simulation_length=self.sim_len,
            model=mod,
            connectivity=self.con,
            coupling=coupling.SigmoidalJansenRit(a=G),
            conduction_speed=np.float64(self.con.speed),
            integrator=self.heunint,
            monitors=(self.mon_eeg,)
        )
        sim.configure()

        [(t, x)] = sim.run()
        return t, x

    def _evaluate(self, X, out, *args, **kwargs):
        pop = X.shape[0]
        F = np.zeros((pop, 2))

        filt_eeg_exp = self.exp_eeg
        exp_signals = filt_eeg_exp.T

        _, _, _, _, band_psd_dict_exp = analyze_populations_with_averaged_psd_bands(
            exp_signals, self.fs, self.window_length, self.overlap,
            self.bands, self.regions_name, self.threshold_ratio
        )

        ratio_exp = band_psd_dict_exp["alpha"]["dominant_psd"] / (
            band_psd_dict_exp["delta"]["dominant_psd"] + band_psd_dict_exp["theta"]["dominant_psd"]
        )

        for i, (G_val, lp_val) in enumerate(X):
            print(f"[eval {i}] G={G_val:.3f}, lp={lp_val:.3f}")
            try:
                teeg, xeeg = self.simulate_TVB(G_val, lp_val)
            except Exception as e:
                print("Simulation failed:", e)
                F[i, :] = [1e6, 1e6]
                continue

            filt_eeg = lowpass_signal(xeeg[1000:, 0, :, :].squeeze(),
                                      fc=self.fc, fs=self.fs, order=self.order)
            sim_signals = filt_eeg.T

            _, _, _, _, band_psd_dict_sim = analyze_populations_with_averaged_psd_bands(
                sim_signals, self.fs, self.window_length, self.overlap,
                self.bands, self.regions_name, self.threshold_ratio
            )

            ratio_sim = band_psd_dict_sim["alpha"]["dominant_psd"] / (
                band_psd_dict_sim["delta"]["dominant_psd"] + band_psd_dict_sim["theta"]["dominant_psd"]
            )

            psd_error = abs(ratio_sim - ratio_exp)

            norm_filt_eeg = (filt_eeg - np.min(filt_eeg)) / (np.max(filt_eeg) - np.min(filt_eeg))
            norm_eeg_exp = (filt_eeg_exp - np.min(filt_eeg_exp)) / (np.max(filt_eeg_exp) - np.min(filt_eeg_exp))

            mean_error = abs(np.mean(norm_filt_eeg) - np.mean(norm_eeg_exp))
            F[i, 0] = psd_error
            F[i, 1] = mean_error

            print(f" -> PSD ratio sim={ratio_sim:.4f}, exp={ratio_exp:.4f}, err={psd_error:.4f}, mean_err={mean_error:.4f}")

        out["F"] = F


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- MAIN CON ARGPARSE -------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def main(pathSC, ngen, npop, type_acquisition, data):
    ## Output directories
    out_dir = './test_output_optim'
    out_dir_img = out_dir + '/img'
    out_dir_data = out_dir + '/data'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_data, exist_ok=True)

    # Used only for the names  - equals for all the subjects
    regions_name = np.loadtxt('/home/neuro_img/Users/Roberta/LOWFIELD_TVB/HC01/regions_labels.txt', dtype=str)

    ## EEG data
    EEG_exp_full = np.loadtxt(data, dtype=float, delimiter=' ')
    exp_eeg = EEG_exp_full[126:, :] #per avere 317000

    ## Simulation parameters
    dt = 2 ** -4
    period_mon_avg = 2 ** -2
    fsamp = 250
    period_mon_eeg = (1 / fsamp) * 1e3
    sim_len = 317000 #circa 6 min

    ## Connectivity and model setup
    con = connectivity.Connectivity.from_file(pathSC)
    con.configure()

    v0 = np.array(6.)
    mod = models.JansenRit(v0=v0)
    sigma = np.zeros(6)
    sigma[4] = mod.a * mod.A * (.320 - .120) * 50e-7
    hiss = noise.Additive(nsig=sigma)
    heunint = integrators.HeunStochastic(dt=dt, noise=hiss)
    my_mon = monitors.TemporalAverage(period=period_mon_eeg)

    ## Optimization setup
    pareto_name = f"pareto_front_tavg_noiseAmato_{npop}_{ngen}_{type_acquisition}.npz"

    params_sim = {
        'SC': pathSC,
        'dt_integrator': dt,
        'period_mon_avg': period_mon_avg,
        'period_mon_eeg': period_mon_eeg,
        'sim_len': sim_len,
        'EEG_data': data,
    }
    df_params = pd.DataFrame.from_dict(params_sim, orient='index')
    df_params.to_csv(out_dir + f"/parameters_{npop}_{ngen}_TRIAL.csv")

    problem = TVBOptimizationProblem(
        exp_eeg=exp_eeg,
        tvb_cfg={"sim_len": sim_len},
        con=con,
        heunint=heunint,
        my_mon=my_mon,
        regions_name=regions_name,
        fsamp=fsamp,
        xl=np.array([0.1, 0.0]),
        xu=np.array([10, 1.0])
    )

    algorithm = NSGA2(pop_size=npop)
    res = minimize(problem, algorithm, ('n_gen', ngen), seed=1, verbose=True)

    print("Optimization finished. Pareto front size:", res.F.shape, res.X.shape)
    np.savez(os.path.join(out_dir_data, pareto_name), X=res.X, F=res.F)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TVB optimization pipeline.")
    parser.add_argument("--pathSC", type=str, required=True,
                        help="Path structural connectivity (es. connectome.zip)")
    parser.add_argument("--ngen", type=int, required=True,
                        help="NSGA number of generation")
    parser.add_argument("--npop", type=int, required=True,
                        help="NSGA number of elements")
    parser.add_argument("--type_acquisition", type=str, required=True,
                        choices=["hyper", "3t", "avg"],
                        help="Type SC")
    parser.add_argument("--data", type=str, required=True,
                        help="Path functional recordings")

    args = parser.parse_args()
    main(args.pathSC, args.ngen, args.npop, args.type_acquisition, args.data)
    
