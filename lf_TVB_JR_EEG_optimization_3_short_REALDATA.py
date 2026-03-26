import matplotlib.pyplot as plt
import numpy as np
from tvb.simulator.lab import *
import pandas as pd
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG

from utils_optim import *
from utils_myoptimizer import *
from utils_mypsd import *


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- PARAMS SET-UP -----------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

## create some output folder
out_dir = './test_output_optim'
out_dir_img = out_dir + '/img'
out_dir_data  = out_dir + '/data'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir_img, exist_ok=True)
os.makedirs(out_dir_data, exist_ok=True)

#root_path = '/media/bcc/Volume/Analysis/Roberta/HCP_TVBmm_30M/106016/T1w/'
#pathSC = root_path + 'SC_dirCB_NOCURATED.zip'

pathSC = '/home/neuro_img/Users/Roberta/LOWFIELD_TVB/HC01/Connectome/Conn_dirCRBL_3T.zip'

regions_name = np.loadtxt('/home/neuro_img/Users/Roberta/LOWFIELD_TVB/HC01/regions_labels.txt', dtype=str)


## EEG
data= '/home/neuro_img/Users/Roberta/LOWFIELD_TVB/HC01/EEG/HC01_WB.txt'
EEG_exp_full = np.loadtxt(data, dtype=float, delimiter=' ') #time x regions
exp_eeg = EEG_exp_full[30000:50000,:]

## simulations -- con questi parametri non ho invalid error in local coupling*y[1]-y[2]
dt = 2 ** -4            #integration steps [ms] -- from tutorial eeg and jansen rit -- usano sempre dt molto piccoli anche 2e-6
period_mon_avg = 2 **-2  #1      #period for time avg monitor [ms]
fsamp =  250          #1e3/1024.0      # 1024 Hz -- from tutorial eeg
period_mon_eeg = (1/fsamp)*1e3  #periond for monitor eeg [ms] -- 0 if not recordings
sim_len = 20000 #320000          #10 sec for testing
ttrans = 1000           #transient time

## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- CONFIGURATION -----------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------

con = connectivity.Connectivity.from_file(pathSC)
con.configure()

## here I left connectivty as default one
#con.speed = np.array(12.5) #setted at 12.5



## model
v0=np.array(6.)
mod = models.JansenRit(v0=v0)

## coupling -- from jansen rit tutorial
#G=np.array(10.0)
#con_coupling = coupling.SigmoidalJansenRit(a=G)

## integration from amato
sigma = np.zeros(6)
sigma[4] = mod.a * mod.A * (.320 - .120) * 50e-7
hiss = noise.Additive(nsig=sigma)
heunint = integrators.HeunStochastic(dt=dt, noise=hiss)

## monitors
my_mon = monitors.TemporalAverage (period= period_mon_eeg)
#my_mon =  monitors.EEG(sensors=sensorsEEG, projection=prEEG, region_mapping=rm, period= period_mon_eeg)
#rec = (mon_tavg, my_mon)


## ---------------------------------------------------------------------------------------------------------------------
## ------------------------------------- TVB SIMULATION ----------------------------------------------------------------
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
                 xl=None,
                 xu=None):
        super().__init__(n_var=2,
                         n_obj=2,
                         xl=xl,
                         xu=xu)

        # fs
        #self.fs = 1000.0 / tvb_cfg["period_mon_eeg"]

        self.sim_len = tvb_cfg["sim_len"]

        # TVB configs
        self.con = con
        self.exp_eeg = exp_eeg
        self.heunint = heunint
        self.mon_eeg = my_mon

        # bands
        bands , _ = get_freqs_bands()
        self.bands = bands

        self.threshold_ratio = 0.0

        self.regions_name = [f"ch{i}" for i in range(len(regions_name))] #[f"ch{i}" for i in range(self.exp_eeg.shape[1])]

        ## params for psd
        self.fs = fsamp #250
        self.window_length = fsamp *2 #500
        self.overlap = 0

        ## params for filtering
        self.fc  = 45
        self.order = 4

    def simulate_TVB(self, global_coupling, lp):

        tau_e, tau_i = update_tau_from_lp(lp)

        a = 1.0 / tau_e
        b = 1.0 / tau_i

        #mod = models.JansenRit(mu=np.array(0.9), v0=np.array(6.0),
        #                       a=np.array([a]), b=np.array([b]))

        mod = models.JansenRit(v0=np.array(6.0),
                               a=np.array([a]), b=np.array([b]))

        G = np.array(global_coupling)

        sim = simulator.Simulator(
            simulation_length=self.sim_len,
            model=mod,
            connectivity=self.con,
            coupling=coupling.SigmoidalJansenRit(a=G),
            conduction_speed=np.float64(self.con.speed),
            integrator=self.heunint,
            monitors= (self.mon_eeg,)
        )
        sim.configure()

        [(t, x)] = sim.run()

        return t, x

    def _evaluate(self, X, out, *args, **kwargs):

        pop = X.shape[0]

        F = np.zeros((pop, 2))

        # PSD ground truth
        #filt_eeg_exp = lowpass_signal(self.exp_eeg[ttrans:, 0, :, :].squeeze(), fc=self.fc, fs=self.fs, order=self.order)
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

            # filtering of EEG sim on var 0
            filt_eeg = lowpass_signal(xeeg[ttrans:, 0, :, :].squeeze(), fc=self.fc, fs=self.fs, order=self.order)

            # preparo segnali (n_regions x time)
            sim_signals = filt_eeg.T

            # PSD bands
            _, _, _, _, band_psd_dict_sim = analyze_populations_with_averaged_psd_bands(
                sim_signals, self.fs, self.window_length, self.overlap,
                self.bands, self.regions_name, self.threshold_ratio
            )

            ratio_sim = band_psd_dict_sim["alpha"]["dominant_psd"] / (
                band_psd_dict_sim["delta"]["dominant_psd"] + band_psd_dict_sim["theta"]["dominant_psd"]
            )

            psd_error = abs(ratio_sim - ratio_exp)

            #normalizzo perchè uno è zscore e l'altro no
            norm_filt_eeg = (filt_eeg - np.min(filt_eeg))/(np.max(filt_eeg)-np.min(filt_eeg))
            norm_eeg_exp = (filt_eeg_exp - np.min(filt_eeg_exp))/(np.max(filt_eeg_exp)-np.min(filt_eeg_exp))

            mean_error = abs(np.mean(norm_filt_eeg) - np.mean(norm_eeg_exp))

            F[i, 0] = psd_error
            F[i, 1] = mean_error

            print(f" -> PSD ratio sim={ratio_sim:.4f}, exp={ratio_exp:.4f}, err={psd_error:.4f}, mean_err={mean_error:.4f}")

        out["F"] = F


pop_size = 8
ngen = 10
type_acq ="3T_muok"

pareto_name = "pareto_front_tavg_noiseAmato_"+str(pop_size)+"_"+str(ngen)+type_acq+".npz"

params_sim = {'SC': pathSC,
              'dt_integrator': dt,
              'periond_mon_avg': period_mon_avg,
              'period_mon_eeg': period_mon_eeg,
              'sim_len': sim_len,
              'EEG_data': data,
              }
df_params = pd.DataFrame.from_dict(params_sim, orient='index')
df_params.to_csv(out_dir+"/parameters"+str(pop_size)+"_"+str(ngen)+".csv")



problem = TVBOptimizationProblem(
    exp_eeg=exp_eeg,
    tvb_cfg={"sim_len": sim_len},
    xl=np.array([0.1, 0.0]),
    xu=np.array([10, 1.0]) #G, lp
)



algorithm = NSGA2(pop_size=pop_size)

res = minimize(problem, algorithm, ('n_gen', ngen), seed=1, verbose=True)

print("Optimization finished. Pareto front size:", res.F.shape, res.X.shape)

np.savez(os.path.join(out_dir_data, pareto_name), X=res.X, F=res.F)
