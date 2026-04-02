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
from tvb.basic.neotraits.api import Final, List, NArray, Attr

from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# Custom libraries
from utils_optim import *
from utils_myoptimizer import *
from utils_mypsd import *


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- JRPSP MODEL ------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

class JRPSP(models.JansenRit):
    """
    Extended Jansen-Rit model with afferent PSP (Amato et al.; https://doi.org/10.1186/s13195-025-01765-z)

    JR standard PLUS two state variables (y6, y7) to model efferent PSP, which sets the JR
    in the correct portion of the phase space for realistic oscillatory activity.

    State variables: y0..y5 (standard JR) + y6, y7 (afferent PSP)
    Coupling variable: y6 (index 6), instead of default y1-y2
    """

    state_variable_range = Final(
        default={
            "y0": np.array([-1.0,    1.0]),
            "y1": np.array([-500.0,  500.0]),
            "y2": np.array([-50.0,   50.0]),
            "y3": np.array([-6.0,    6.0]),
            "y4": np.array([-20.0,   20.0]),
            "y5": np.array([-500.0,  500.0]),
            "y6": np.array([-20.0,   20.0]),
            "y7": np.array([-500.0,  500.0]),
        }
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
        default=("y0", "y1", "y2", "y3"),
    )

    state_variables = tuple("y0 y1 y2 y3 y4 y5 y6 y7".split())
    _nvar = 8
    cvar  = np.array([6], dtype=np.int32)   # coupling via y6

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        dy = np.zeros((8, state_variables.shape[1], 1))
        # Standard JR equations for y0..y5
        dy[:6] = super().dfun(state_variables[:6], coupling, local_coupling)
        # PSP extension for efferent signal (eq. 8 in Amato et al.)
        y0, y1, y2, y3, y4, y5, y6, y7 = state_variables
        a_d = self.a / 3.0
        sigm_y1_y2 = 2.0 * self.nu_max / (
            1.0 + np.exp(self.r * (self.v0 - (y1 - y2)))
        )
        dy[6] = y7
        dy[7] = self.A * a_d * sigm_y1_y2 - 2.0 * a_d * y7 - (self.a ** 2) * y6
        return dy


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- TAU UPDATE FROM lp -----------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def update_tau_from_lp_DADD(lp_val):
    """
    Computes a=1/tau_e and b=1/tau_i from lp following (Amato et al.; https://doi.org/10.1186/s13195-025-01765-z):
        tau_i -> tau_i_HC + lp * (tau_i_max - tau_i_HC)
        tau_e -> tau_e_HC + lp * (tau_e_min - tau_e_HC)

    Expressed directly in terms of a and b (TVB parameters):
        a -> a_HC + lp * (a_max - a_HC)   with a_HC=0.100, a_max=0.112
        b -> b_HC + lp * (b_min - b_HC)   with b_HC=0.050, b_min=0.025

    lp=0 -> healthy baseline  (a=0.100, b=0.050)
    lp=1 -> most degenerated  (a=0.112, b=0.025)

    Returns: a_val (1/tau_e), b_val (1/tau_i)
    """
    a_val = 0.100 + lp_val * (0.112 - 0.100)
    b_val = 0.050 + lp_val * (0.025 - 0.050)
    return float(a_val), float(b_val)


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- FC UTILITIES -----------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def compute_fc_score(signal_2d, threshold_ratio=0.13):
    """
    Computes the scalar FC score used in DADD grid search:
      1. Pearson correlation matrix (n_regions x n_regions)
      2. Binary threshold at threshold_ratio * max(corr)
      3. Returns mean of thresholded matrix

    Parameters
    ----------
    signal_2d : np.ndarray, shape (time, n_regions)
    threshold_ratio : float - Fixed at 0.13 based on Amato et al.; https://doi.org/10.1186/s13195-025-01765-z

    Returns
    -------
    float : mean of thresholded FC matrix
    """
    corr = np.corrcoef(signal_2d.T)                       # (n_regions, n_regions)
    corr_th = corr > threshold_ratio * np.max(corr)
    return float(corr_th.mean())


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- STANDALONE WORKER FUNCTION ---------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def evaluate_single(args):
    """
    Evaluate function for a single individual.
    args = only scalars and paths -- TVB objects are non-picklable,
    so each worker builds its own TVB objects.

    Objectives (aligned to DADD loss function):
      F[0] = |ratio_alpha_sim - ratio_alpha_exp|    (PSD relative alpha power)
      F[1] = |fc_score_sim   - fc_score_exp|        (mean thresholded FC)

      - Coupling  : Sigmoidal(a=10.0) fixed; G scales SC weights for biological interpretability of the optimization
      - Velocity  : np.inf  (no conduction delays)
      - Noise     : nsig shape (8,) for JRPSP; nsig[4] = A * a_current * (0.320-0.120) * noise_base
      - Monitor   : TemporalAverage only (tavg = proxy for LFP for Jansen-Rit model)
      - PSD ratio : alpha / total  (not alpha / delta+theta)
    """

    (G_val, lp_val,
     pathSC_w, sim_len_w, ttrans_w,
     dt_w, noise_base_w, period_mon_w,
     fsamp_w, window_length_w, overlap_w,
     fc_w, order_w,
     bands_w, regions_name_w, threshold_ratio_w,
     ratio_alpha_exp_w, fc_score_exp_w) = args

    try:
        # Each worker builds its own connectivity (TVB not picklable)
        con_w = connectivity.Connectivity.from_file(pathSC_w)
        con_w.configure()

        #scale SC weights by G, use Sigmoidal(a=10.0) fixed
        con_w.weights *= G_val

        # lp -> a, b
        a_val, b_val = update_tau_from_lp_DADD(lp_val)


        # JRPSP model with same fixed parameters as Amato et al.; https://doi.org/10.1186/s13195-025-01765-z

        model_w = JRPSP(
            A      = np.array([3.25]),
            B      = np.array([22.0]),
            v0     = np.array([6.0]),
            a      = np.array([a_val]),
            b      = np.array([b_val]),
            r      = np.array([0.56]),
            nu_max = np.array([0.0025]),
            J      = np.array([128.0]),
            a_1    = np.array([1.0]),
            a_2    = np.array([0.8]),
            a_3    = np.array([0.25]),
            a_4    = np.array([0.25]),
            mu     = np.array([0.22]),
            variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
        )

        # Noise: shape (8,) for JRPSP; only component 4 is driven
        # nsig[4] = A * a_current * (0.320 - 0.120) * noise_base
        nsig_w    = np.zeros(8)
        nsig_w[4] = 3.25 * a_val * (0.320 - 0.120) * noise_base_w
        hiss_w    = noise.Additive(nsig=nsig_w)
        heunint_w = integrators.HeunStochastic(dt=dt_w, noise=hiss_w)

        # Let's start the simulation!
        mon_w = monitors.TemporalAverage(period=period_mon_w)

        sim = simulator.Simulator(
            simulation_length = sim_len_w,
            model             = model_w,
            connectivity      = con_w,
            coupling          = coupling.Sigmoidal(a=np.array([10.0])),
            conduction_speed  = np.inf,
            integrator        = heunint_w,
            monitors          = (mon_w,)
        )
        sim.configure()
        [(t, x)] = sim.run()

        # x shape: (time, n_vars, n_regions, 1)
        tavg = x[ttrans_w:, 0, :, 0]   # (time, n_regions)

        # F[0]: relative alpha PSD error  (= alpha / total)

        sim_signals = tavg.T   # (n_regions, time) as expected by my library utils

        _, _, _, _, band_psd_dict_sim = analyze_populations_with_averaged_psd_bands(
            sim_signals, fsamp_w, window_length_w, overlap_w,
            bands_w, regions_name_w, threshold_ratio_w
        )

        # Try use band_power (sum over frequencies in band), instead of dominant_psd (peak).
        alpha_psd = band_psd_dict_sim["alpha"]["band_power"]
        total_psd = sum(
            band_psd_dict_sim[b]["band_power"]
            for b in band_psd_dict_sim
        )
        ratio_alpha_sim = alpha_psd / total_psd if total_psd > 0 else 0.0

        psd_error = abs(ratio_alpha_sim - ratio_alpha_exp_w)

        # F[1]: FC score error
        fc_score_sim = compute_fc_score(tavg, threshold_ratio=0.13)
        fc_error     = abs(fc_score_sim - fc_score_exp_w)

        print(f"   G={G_val:.3f}, lp={lp_val:.3f} -> "
              f"psd_err={psd_error:.4f}, fc_err={fc_error:.4f}", flush=True)

        return float(psd_error), float(fc_error)

    except Exception as e:
        print(f"   Simulation FAILED G={G_val:.3f}, lp={lp_val:.3f}: {e}", flush=True)
        return 1e6, 1e6


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- TVB OPTIMIZATION PROBLEM -----------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

class TVBOptimizationProblem(Problem, ABC):
    """
    2-variable problem: [G, lp]

    Objectives (DADD-aligned):
      F[:,0] = relative alpha PSD error   |ratio_alpha_sim - ratio_alpha_exp|
      F[:,1] = FC score error             |fc_score_sim   - fc_score_exp|
    """

    def __init__(self, exp_eeg, tvb_cfg, pathSC, regions_name,
                 fsamp, n_jobs, xl=None, xu=None):

        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)

        self.sim_len    = tvb_cfg["sim_len"]
        self.ttrans     = tvb_cfg["ttrans"]
        self.dt         = tvb_cfg["dt"]
        self.noise_base = tvb_cfg["noise_base"]
        self.period_mon = tvb_cfg["period_mon"]
        self.pathSC     = pathSC
        self.n_jobs     = n_jobs

        self.bands, _        = get_freqs_bands()
        self.threshold_ratio = 0.0
        self.regions_name    = [f"ch{i}" for i in range(len(regions_name))]

        self.fs            = fsamp
        self.window_length = fsamp * 2
        self.overlap       = 0
        self.fc_filt       = 45
        self.order         = 4

        # Pre-compute experimental biomarkers (done once at init)
        # exp_eeg shape: (time, n_regions) -- sLORETA region space

        exp_signals = exp_eeg.T   # (n_regions, time) for utils

        _, _, _, _, band_psd_dict_exp = analyze_populations_with_averaged_psd_bands(
            exp_signals, self.fs, self.window_length, self.overlap,
            self.bands, self.regions_name, self.threshold_ratio
        )

        alpha_psd_exp = band_psd_dict_exp["alpha"]["band_power"]
        total_psd_exp = sum(
            band_psd_dict_exp[b]["band_power"]
            for b in band_psd_dict_exp
        )
        self.ratio_alpha_exp = alpha_psd_exp / total_psd_exp if total_psd_exp > 0 else 0.0

        # FC score on experimental region-space signal
        self.fc_score_exp = compute_fc_score(exp_eeg, threshold_ratio=0.13)

        print(f"[init] ratio_alpha_exp={self.ratio_alpha_exp:.4f}, "
              f"fc_score_exp={self.fc_score_exp:.4f}", flush=True)

    def _evaluate(self, X, out, *args, **kwargs):

        args_list = [
            (float(G_val), float(lp_val),
             self.pathSC, self.sim_len, self.ttrans,
             self.dt, self.noise_base, self.period_mon,
             self.fs, self.window_length, self.overlap,
             self.fc_filt, self.order,
             self.bands, self.regions_name, self.threshold_ratio,
             self.ratio_alpha_exp, self.fc_score_exp)
            for G_val, lp_val in X
        ]

        # parallelization
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(evaluate_single, args_list)

        out["F"] = np.array(results)


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- PLOT CALLBACK ----------------------------------------------------------------
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

        self.hv_history      = []
        self.min_psd_history = []
        self.min_fc_history  = []

    def notify(self, algorithm):

        gen = algorithm.n_gen
        F   = algorithm.opt.get("F")

        if self.ref_point is None:
            all_F = algorithm.pop.get("F")
            self.ref_point = np.max(all_F, axis=0) * 1.1
            print(f"[callback] ref_point = {self.ref_point}", flush=True)

        hv_val = HV(ref_point=self.ref_point)(F)

        self.gen_history.append(gen)
        self.hv_history.append(hv_val)
        self.min_psd_history.append(np.min(F[:, 0]))
        self.min_fc_history.append(np.min(F[:, 1]))

        print(f"[gen {gen:03d}] Pareto size={len(F)}, HV={hv_val:.6f}, "
              f"min_psd={self.min_psd_history[-1]:.4f}, "
              f"min_fc={self.min_fc_history[-1]:.4f}", flush=True)

        if gen % self.PLOT_EVERY == 0:
            self._save_plot(gen, F)

    def _save_plot(self, gen, F):

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.scatter(F[:, 0], F[:, 1], c='steelblue', s=60,
                   edgecolors='k', linewidths=0.5)
        ax.set_xlabel('Relative alpha PSD error', fontsize=12)
        ax.set_ylabel('FC score error', fontsize=12)
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
                 label='min alpha PSD error')
        ax3.plot(self.gen_history, self.min_fc_history,
                 marker='s', color='tomato', linewidth=1.8, markersize=5,
                 label='min FC error')
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
## ------------------------------------- MAIN -------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

def main(pathSC, ngen, npop, ncpu, type_acquisition, data):

    root_path    = './'
    out_dir      = os.path.join(root_path, 'test_output_optim', f'{type_acquisition}_ngen{ngen}_npop{npop}_ncpu{ncpu}_data')
    out_dir_img  = os.path.join(out_dir, 'img')
    out_dir_data = os.path.join(out_dir, 'data')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_data, exist_ok=True)

    n_jobs = ncpu

    regions_name = np.loadtxt(os.path.join(root_path, 'LOWFIELD_TVB/HC01/regions_labels.txt'), dtype=str)
-
    # Load experimental data (region-based, sLORETA source space)
    EEG_exp_full = np.loadtxt(data, dtype=float, delimiter=' ')
    exp_eeg      = EEG_exp_full[126:, :]   # (time, n_regions)

    # Simulation parameters
    dt             = 2 ** -4
    period_mon_avg = 2 ** -2
    fsamp          = 250
    period_mon     = (1 / fsamp) * 1e3   # ms
    sim_len        = 317000
    ttrans         = 1000

    # noise_base: scalar multiplier:
    # nsig[4] = A * a_current * (0.320 - 0.120) * noise_base
    # A=3.25, a_HC=0.100, range=(0.320-0.120)=0.200
    # noise_base=50e-3 is the value used in Amato et al. (final simulations)
    noise_base = 50e-3

    tvb_cfg = {
        'sim_len':   sim_len,
        'ttrans':    ttrans,
        'dt':        dt,
        'noise_base': noise_base,
        'period_mon': period_mon,
    }

    pareto_name = f"pareto_front_GEN{ngen}_POP{npop}_CPU{ncpu}_{type_acquisition}.npz"
    params_sim  = {
        'SC':             pathSC,
        'dt_integrator':  dt,
        'period_mon':     period_mon,
        'sim_len':        sim_len,
        'noise_base':     noise_base,
        'EEG_data':       data,
        'n_jobs':         n_jobs,
        'model':          'JRPSP',
        'coupling':       'Sigmoidal(a=10.0)',
        'velocity':       'np.inf',
    }
    pd.DataFrame.from_dict(params_sim, orient='index').to_csv(os.path.join(out_dir, f"parameters_GEN{ngen}_POP{npop}_CPU{ncpu}_{type_acquisition}_DADDlike.csv"))

    # ------------------------------------------------------------------
    # Problem + algorithm + callback
    # G in [0.1, 10], lp in [0, 1]  (same bounds as before)
    # ------------------------------------------------------------------
    problem = TVBOptimizationProblem(
        exp_eeg      = exp_eeg,
        tvb_cfg      = tvb_cfg,
        pathSC       = pathSC,
        regions_name = regions_name,
        fsamp        = fsamp,
        n_jobs       = n_jobs,
        xl           = np.array([0.1, 0.0]),
        xu           = np.array([20.0, 1.0]) #since the max explored in Amato et al.; https://doi.org/10.1186/s13195-025-01765-z is G=10, I put 20
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

    np.savez(os.path.join(out_dir_data, pareto_name), X=res.X, F=res.F)
    callback._save_plot(ngen, res.F)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TVB parallelized optimization pipeline (DADD-aligned).")

    parser.add_argument("--pathSC", type=str, required=True,
                        help="Name of SC file (e.g. connectome.zip)")
    parser.add_argument("--ngen", type=int, required=True,
                        help="Number of generations in NSGA2")
    parser.add_argument("--npop", type=int, required=True,
                        help="Population size in NSGA2")
    parser.add_argument("--ncpu", type=int, required=True,
                        help="Number of CPUs (should equal npop for max efficiency)")
    parser.add_argument("--type_acquisition", type=str, required=True,
                        choices=["hyper", "3t", "avg"],
                        help="DWI acquisition type")
    parser.add_argument("--data", type=str, required=True,
                        help="Full path of region-space EEG data (sLORETA)")

    args = parser.parse_args()
    main(args.pathSC, args.ngen, args.npop, args.ncpu, args.type_acquisition, args.data)
