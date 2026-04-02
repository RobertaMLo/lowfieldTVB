import matplotlib
matplotlib.use('Agg')  # backend NON-interactive needed for SLURM (i.e., no display)
import matplotlib.pyplot as plt

import os
import time
import argparse
from abc import ABC
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tvb.simulator.lab import *

from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# # Custom libraries
from utils_optim import *
from utils_myoptimizer import *
from utils_mypsd import *


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- STANDALONE WORKER FUNCTION ----------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def evaluate_single(args):
    """
    Evaluate function for a single element.
    args = only scalar and path -- TVB is non pickable, so it is better that each worker (cpu) built its own TVB objects
    """
    
    (G_val, lp_val,
     pathSC_w, sim_len_w, ttrans_w,
     dt_w, sigma_val_w, period_mon_eeg_w,
     fsamp_w, window_length_w, overlap_w,
     fc_w, order_w,
     bands_w, regions_name_w, threshold_ratio_w,
     ratio_exp_w, norm_eeg_exp_mean_w) = args

    try:
        # for each process I set the connectivity, just to be sure. TVB is non pickable -- it is not huge amount of time.
        con_w = connectivity.Connectivity.from_file(pathSC_w)
        con_w.configure()
        print(f"CONDUCION SPEED={con_w.speed}" ,flush=True)

        tau_e, tau_i = update_tau_from_lp(lp_val)
        a = 1.0 / tau_e
        b = 1.0 / tau_i

        mod_w = models.JansenRit(mu=np.array(0.9), v0=np.array(6.0),
                                  a=np.array([a]), b=np.array([b]))

        sigma_w = np.zeros(6)
        sigma_w[4] = sigma_val_w
        hiss_w    = noise.Additive(nsig=sigma_w)
        heunint_w = integrators.HeunStochastic(dt=dt_w, noise=hiss_w)
        mon_w     = monitors.TemporalAverage(period=period_mon_eeg_w)

        sim = simulator.Simulator(
            simulation_length=sim_len_w,
            model=mod_w,
            connectivity=con_w,
            coupling=coupling.SigmoidalJansenRit(a=np.array(G_val)),
            conduction_speed=float(con_w.speed),
            integrator=heunint_w,
            monitors=(mon_w,)
        )
        sim.configure()
        [(t, x)] = sim.run()

        #PRIMA ERA COSi:conduction_speed=np.float64(con_w.speed),

        filt_eeg    = lowpass_signal(x[ttrans_w:, 0, :, :].squeeze(),
                                     fc=fc_w, fs=fsamp_w, order=order_w)
        sim_signals = filt_eeg.T

        _, _, _, _, band_psd_dict_sim = analyze_populations_with_averaged_psd_bands(
            sim_signals, fsamp_w, window_length_w, overlap_w,
            bands_w, regions_name_w, threshold_ratio_w
        )

        ratio_sim = band_psd_dict_sim["alpha"]["dominant_psd"] / (
            band_psd_dict_sim["delta"]["dominant_psd"] +
            band_psd_dict_sim["theta"]["dominant_psd"]
        )

        psd_error = abs(ratio_sim - ratio_exp_w)

        norm_filt_eeg = (filt_eeg - np.min(filt_eeg)) / (np.max(filt_eeg) - np.min(filt_eeg))
        mean_error    = abs(np.mean(norm_filt_eeg) - norm_eeg_exp_mean_w)

        print(f"   G={G_val:.3f}, lp={lp_val:.3f} -> "
              f"psd_err={psd_error:.4f}, mean_err={mean_error:.4f}", flush=True)

        return float(psd_error), float(mean_error)

    except Exception as e:
        print(f"   Simulation FAILED G={G_val:.3f}, lp={lp_val:.3f}: {e}", flush=True)
        return 1e6, 1e6


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- TVB OPTIMIZATION PROBLEM ------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

class TVBOptimizationProblem(Problem, ABC):
    """
    Problem con 2 variabili: [G, lp]
    Obiettivi:
      F[:,0] = errore PSD ratio (alpha / delta+theta)
      F[:,1] = errore media segnale
    """

    def __init__(self, exp_eeg, tvb_cfg, pathSC, regions_name,
                 fsamp, n_jobs, xl=None, xu=None):

        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)

        self.sim_len        = tvb_cfg["sim_len"]
        self.ttrans         = tvb_cfg["ttrans"]
        self.dt             = tvb_cfg["dt"]
        self.sigma_val      = tvb_cfg["sigma_val"]
        self.period_mon_eeg = tvb_cfg["period_mon_eeg"]
        self.pathSC         = pathSC
        self.n_jobs         = n_jobs

        self.bands, _        = get_freqs_bands()
        self.threshold_ratio = 0.0
        self.regions_name    = [f"ch{i}" for i in range(len(regions_name))]

        self.fs            = fsamp
        self.window_length = fsamp * 2
        self.overlap       = 0
        self.fc            = 45
        self.order         = 4

        # Elaboration on experimental data --> done only once (when class is instanced)
        exp_signals = exp_eeg.T

        _, _, _, _, band_psd_dict_exp = analyze_populations_with_averaged_psd_bands(
            exp_signals, self.fs, self.window_length, self.overlap,
            self.bands, self.regions_name, self.threshold_ratio
        )

        self.ratio_exp = band_psd_dict_exp["alpha"]["dominant_psd"] / (
            band_psd_dict_exp["delta"]["dominant_psd"] +
            band_psd_dict_exp["theta"]["dominant_psd"]
        )

        exp_norm = (exp_eeg - np.min(exp_eeg)) / (np.max(exp_eeg) - np.min(exp_eeg))
        self.norm_eeg_exp_mean = float(np.mean(exp_norm))

        print(f"[init] ratio_exp={self.ratio_exp:.4f}, "
              f"norm_eeg_exp_mean={self.norm_eeg_exp_mean:.4f}", flush=True)

    def _evaluate(self, X, out, *args, **kwargs):

        ## args list needed as input by evaluate function!
        ## Mi serve un elemento per individuo
        args_list = [
            (float(G_val), float(lp_val),
             self.pathSC, self.sim_len, self.ttrans,
             self.dt, self.sigma_val, self.period_mon_eeg,
             self.fs, self.window_length, self.overlap,
             self.fc, self.order,
             self.bands, self.regions_name, self.threshold_ratio,
             self.ratio_exp, self.norm_eeg_exp_mean)
            for G_val, lp_val in X
        ]

        # # HERE I AM DOING PARALLEL!!!!!!!
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(evaluate_single, args_list)
            #pool = gestore dei processi. Dispacth processi a cpu
            #.map = mappa il processo ai core

        out["F"] = np.array(results)   # shape (pop_size, 2)


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- PLOT FOR EVALUATION -------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

class PlotCallback(Callback):
    """
    Saves a PNG every PLOT_EVERY generations with 3 panels:
      - panel 1: current Pareto front scatter
      - panel 2: hypervolume curve across generations
      - panel 3: minimum of each objective across generations
    """

    PLOT_EVERY = 3

    def __init__(self, out_dir_img):
        super().__init__()
        self.out_dir_img = out_dir_img
        self.gen_history = []
        self.ref_point   = None

        self.hv_history       = []
        self.min_psd_history  = []   # min(psd_error) per generation
        self.min_mean_history = []   # min(mean_error) per generation

    def notify(self, algorithm):

        gen = algorithm.n_gen
        F   = algorithm.opt.get("F")

        # set reference point at generation 1 from worst observed values
        if self.ref_point is None:
            all_F = algorithm.pop.get("F")
            self.ref_point = np.max(all_F, axis=0) * 1.1
            print(f"[callback] ref_point = {self.ref_point}", flush=True)

        hv_val = HV(ref_point=self.ref_point)(F)

        self.gen_history.append(gen)
        self.hv_history.append(hv_val)
        self.min_psd_history.append(np.min(F[:, 0]))
        self.min_mean_history.append(np.min(F[:, 1]))

        print(f"[gen {gen:03d}] Pareto size={len(F)}, HV={hv_val:.6f}, "
              f"min_psd={self.min_psd_history[-1]:.4f}, "
              f"min_mean={self.min_mean_history[-1]:.4f}", flush=True)

        if gen % self.PLOT_EVERY == 0:
            self._save_plot(gen, F)

    def _save_plot(self, gen, F):

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.scatter(F[:, 0], F[:, 1], c='steelblue', s=60,
                   edgecolors='k', linewidths=0.5)
        ax.set_xlabel('PSD ratio error', fontsize=12)
        ax.set_ylabel('Mean signal error', fontsize=12)
        ax.set_title(f'Pareto front — generation {gen}', fontsize=13)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(self.gen_history, self.hv_history,
                 marker='o', color='darkorange', linewidth=1.8, markersize=5)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Hypervolume', fontsize=12)
        ax2.set_title('Hypervolume per generation', fontsize=13)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(self.gen_history, self.min_psd_history,
                 marker='o', color='steelblue', linewidth=1.8, markersize=5,
                 label='min PSD ratio error')
        ax3.plot(self.gen_history, self.min_mean_history,
                 marker='s', color='tomato', linewidth=1.8, markersize=5,
                 label='min mean error')
        ax3.set_xlabel('Generation', fontsize=12)
        ax3.set_ylabel('Minimum objective on front', fontsize=12)
        ax3.set_title('Objective minima per generation', fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(self.out_dir_img, f'pareto_gen_{gen:03d}.png')
        plt.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"[callback] figure saved: {fname}", flush=True)


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- MAIN -------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def main(pathSC, ngen, npop, ncpu, type_acquisition, data):

    root_path    = './'
    out_dir      = os.path.join(root_path, 'test_output_optim',f'{type_acquisition}_data')
    out_dir_img  = os.path.join(out_dir, 'img')
    out_dir_data = os.path.join(out_dir, 'data')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_data, exist_ok=True)

    ## ntasks or ncpupertasks in SLURM (depend on cluster)
    n_jobs = ncpu

    regions_name = np.loadtxt(
        os.path.join(root_path, 'LOWFIELD_TVB/HC01/regions_labels.txt'), dtype=str
    )

    ## EEG
    EEG_exp_full = np.loadtxt(data, dtype=float, delimiter=' ')
    exp_eeg      = EEG_exp_full[126:, :]   # ~317000 samples

    ## sim params
    dt             = 2 ** -4
    period_mon_avg = 2 ** -2
    fsamp          = 250
    period_mon_eeg = (1 / fsamp) * 1e3
    sim_len        = 317000
    ttrans         = 1000

    ## sigma as a scalar
    sigma_val = 0.1 * 3.25 * (.320 - .120) * 50e-7

    tvb_cfg = {
        'sim_len':        sim_len,
        'ttrans':         ttrans,
        'dt':             dt,
        'sigma_val':      sigma_val,
        'period_mon_eeg': period_mon_eeg,
    }

    ## salva parametri run
    pareto_name = f"pareto_front_tavg_noiseAmato_{npop}_{ngen}_{type_acquisition}.npz"
    params_sim  = {
        'SC':             pathSC,
        'dt_integrator':  dt,
        'period_mon_avg': period_mon_avg,
        'period_mon_eeg': period_mon_eeg,
        'sim_len':        sim_len,
        'EEG_data':       data,
        'n_jobs':         n_jobs,
    }
    pd.DataFrame.from_dict(params_sim, orient='index').to_csv(
        os.path.join(out_dir, f"parameters_{npop}_{ngen}_{type_acquisition}_TRIAL.csv")
    )

    ## problema + algoritmo + callback
    problem = TVBOptimizationProblem(
        exp_eeg=exp_eeg,
        tvb_cfg=tvb_cfg,
        pathSC=pathSC,
        regions_name=regions_name,
        fsamp=fsamp,
        n_jobs=n_jobs,
        xl=np.array([0.1, 0.0]),
        xu=np.array([10,  1.0])   # bounds: G in [0.1, 10], lp in [0, 1]
    )

    algorithm = NSGA2(pop_size=npop)
    callback  = PlotCallback(out_dir_img=out_dir_img)

    start_time = time.time()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', ngen),
        seed=1,
        verbose=True,
        callback=callback
    )

    elapsed = time.time() - start_time
    print(f"\nOptimization finished. Pareto front size: {res.F.shape}, {res.X.shape}")
    print(f"Tempo totale: {elapsed:.1f} s ({elapsed/60:.1f} min)", flush=True)

    ## Final Pareto
    np.savez(os.path.join(out_dir_data, pareto_name), X=res.X, F=res.F)

    ## plot finale indipendente da PLOT_EVERY
    callback._save_plot(ngen, res.F)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TVB parallelized optimization pipeline.")

    parser.add_argument("--pathSC", type=str, required=True,
                        help="Name of SC (es. connectome.zip)")
    parser.add_argument("--ngen", type=int, required=True,
                        help="Number of gen in NSGA2 algorithm")
    parser.add_argument("--npop", type=int, required=True,
                        help="Number of individuals in a population of NSGA2 algorithm")
    parser.add_argument("--ncpu", type=int, required=True,
                        help="should be = npop for maximize the efficiency")
    parser.add_argument("--type_acquisition", type=str, required=True,
                        choices=["hyper", "3t", "avg"],
                        help="DWI acuisition type")
    parser.add_argument("--data", type=str, required=True,
                        help="full path of EEG data")

    args = parser.parse_args()
    main(args.pathSC, args.ngen, args.npop, args.ncpu, args.type_acquisition, args.data)
