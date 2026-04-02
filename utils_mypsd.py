"""
Functions to compute the PSD and quantified the contribution of each functional band
author: robertalorenzi - v.01

Refs.:
Lorenzi, R.M., Palesi, F., Casellato, C. et al. Region-specific mean field models enhance simulations of
local and global brain dynamics.
npj Syst Biol Appl 11, 66 (2025). https://doi.org/10.1038/s41540-025-00543-9
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

def set_params(fs=250, window_length=250, noverlap=0, threshold_ratio=0):
    """
    Function to set the parameters of the PSD.
    fs = float = sampling frequency
    window_length = int = number of samples, fs/window_length=resolution - here 1 Hz
    noverlap = int = noverlap parameter - usually set as window_length // 2 i.e. 50% overlap
    threshold_ratio = float = threshold ratio to select the psd peak

    ** Default parameters are set base ond multimodal TVB - under development **
    """

    dict_psd={}

    dict_psd['fs'] = fs
    dict_psd['window_length'] = window_length
    dict_psd['noverlap'] = noverlap
    dict_psd['threshold_ratio'] = threshold_ratio

    return dict_psd


def get_freqs_bands(gamma_max=45):
    """
    Function to get the range for each frequency band
    return: dictionary
    """
    bands = {
    'delta': (0, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 25),
    'gamma': (25, gamma_max)
    }

    colors_band = {
        'delta': 'red',
        'theta': 'green',
        'alpha': 'blue',
        'beta': 'purple',
        'gamma': 'orange'
    }

    return bands, colors_band


def lowpass_signal(signal, fc, fs, order):
    """
    Function to filter the signal applying a low poass
    Input:
        signal          (narray) = data on which the FFT is performed  [nvar x timepoints]
        fc              (int)    = cut frequency
        fs              (int)    = sampling frequency

    Output:
        filtered_signal (array) = filtered signal [nvar, timepoints]
    """

    b, a = butter(order, fc, btype='low', fs=fs)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def compute_psd(signal, fs, nperseg, noverlap, verbose=False):
    """
    Function compute the psd using welch method for FFT to reduce variance and improve Singal to Noise Ration.
    Signal is divided into segments with the same length using a windowing methods
    ensuring a certain overlap between each pair of asjacent segment.
    FFT is computed for each segment; PSD of each segment is averaged getting one PSD.
    This reduces the variance of the PSD.

    Input:
        signal          (narray) = data on which the FFT is performed  [nvar x timepoints]
        fr              (int)    = sampling frequency choosen according to Nyquist theorem
        nperseg         (int)    = length of the segment. Larger = ++freq_res --stability
        noverlap        (int)    = number of samples in each segment overlapping with previous segment.
                                    typical choice = 50% to 75 % of nperseg.
                                    ++ overlap = --variance, ++ computational costs

    Output:
        <pop_name>_fr_ar     (array) = avg firing rate for <pop> [nsubj, timepoints]
        <pop_name>_fr_ar_sd  (array) = sd firing rate for <pop> [nsubj, timepoints]

    """

    if verbose:
        print('Debug inside psd computation:\n fs', fs, '\nnperseg', nperseg, '\noverlap', noverlap)
        print('checking signals again: ', np.shape(signal))

    freqs_all, psd_all = welch(signal,
                               fs=fs,
                               nperseg=nperseg,
                               noverlap=noverlap,
                               detrend='constant', #instead of doing signal-mean; nfft = 500 not required
                               )

    #normalizing psd and removing highest frequencies
    #psd = psd_all[:int(len(psd_all)*0.7)]
    #freqs  = freqs_all[:int(len(psd_all)*0.7)

    freqs  = freqs_all
    #psd = psd_all/np.max(psd_all)
    psd = psd_all

    idx_freq_max = np.argmax(psd)
    #print('Sono il massimo in Hz: ', freqs[idx_freq_max])

    return freqs, psd


def compute_auc_and_significant_counts(freqs, psd, bands, threshold_ratio):
    """
    Function to compute holisitc quantitative scores of PSD for each band:
    - AUC using trapz method
    - Frequency densitiy = #freq above a threshold

    Input:
        freqs           (array) = frequency output of welch method
        psd             (array) = psd output of welch method
        bands           (dict)  = <band_name>: (low, high) frequency bands of iterest
        threshold_ratio (float) = significant frequency (here 0.2)

    Output:
        auc_counts      (dict) = auc score and how many supra-threshold frequencies for each band
    """
    max_psd = np.max(psd)
    threshold = max_psd * threshold_ratio

    auc_counts = {band: {'auc': 0, 'significant_count': 0} for band in bands}

    # doing the computation for each band separately
    for band, (low, high) in bands.items():

        # splitting the psd for each band
        band_mask = (freqs >= low) & (freqs <= high)
        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]

        #AUC
        auc = np.trapezoid(band_psd, band_freqs) #numpy v = 2 # before: np.trapz(band_psd, band_freqs)

        # Frequency density
        significant_count = np.sum(band_psd > threshold)

        auc_counts[band]['auc'] = auc
        auc_counts[band]['significant_count'] = significant_count

    return auc_counts


def find_dominant_frequency(freqs, psd, bands):
    """
    Function to compute the maximum frequency for each band.
    "Classic" apprach to get the band-specific carrier frequency

    Input:
        freqs                   (array) = frequency output of welch method
        psd                     (array) = psd output of welch method
        bands                   (dict)  = <band_name>: (low, high) frequency bands of iterest

    Output:
        dominant_frequencies   (dict) = carrier frequency and corresponding psd value
    """
    dominant_frequencies = {band: {'dominant_freq': 0, 'dominant_psd': 0} for band in bands}

    for band, (low, high) in bands.items():

        band_mask = (freqs > low) & (freqs <= high)
        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]

        dominant_freq = band_freqs[np.argmax(band_psd)]
        dominant_psd = band_psd[np.argmax(band_psd)]

        dominant_frequencies[band]['dominant_freq'] = dominant_freq
        dominant_frequencies[band]['dominant_psd'] = dominant_psd

        #print(band)
        #print(dominant_frequencies[band]['dominant_freq'], dominant_frequencies[band]['dominant_psd'])

    return dominant_frequencies


def analyze_populations_with_averaged_psd_bands(signals, fs, window_length, noverlap, bands, regions_name, threshold_ratio, norm = False):
    """
    Routine to compute:
    - AUC and frequency density
    - dominant_frequency
    - averaged PSD
    - PSD averaged per band

    Input:
        signals: shape (n_regions, n_timepoints) .N.B. n_regions can be also n_subjects!!! It depends on the file construction.
        Here signal is nregions x timepoint and it is intended to run recursively on different subjects.

    Output:
        band_psd_dict: {
            'delta': {'freqs_max': <float>, 'psd_peak': <float>},
            ...
        }
    """

    nregions, timepoints = signals.shape

    psd_all = []
    #auc_counts_all = []
    #dominant_freqs_all = []

    for region_idx in range(nregions):

        #reg_name = regions_name[region_idx]
        #print(reg_name)

        # sept 8 2025 - check the function
        # for each region compute psd baby it is the main ingredient

        freqs, psd = compute_psd(signals[region_idx], fs, window_length, noverlap)

        if norm:
            psd = psd/np.max(psd)

        # for each region compute the auc
        #auc_counts = compute_auc_and_significant_counts(freqs, psd, bands, threshold_ratio)

        # for each region compute the dominant frequency -- consder to replace with find_peaks
        #dominant_freqs = find_dominant_frequency(freqs, psd, bands)


        #auc_counts_all.append(auc_counts)
        #dominant_freqs_all.append(dominant_freqs)
        psd_all.append(psd)

        #print('\n')

    # qui lavoro con psd che è nreg x timepoints -
    # isolo le bande di frequenze e per ogni banda faccio la media sulle regioni
    # mi calcolo max psd e freq sulla media su regioni per banda - salvo in dizionario

    psd_all_arr = np.array(psd_all)
    band_psd_dict = {}
    for band, (low, high) in bands.items():

        # for each band mi prendo le frequenze limite
        band_mask = (freqs >= low) & (freqs < high) #high not included
        # isolo freqs di psd per quella banda
        band_freqs = freqs[band_mask]
        # isolo il psd per quella banda
        band_psds = psd_all_arr[:, band_mask]  # shape: (n_regions, n_freqs_in_band)
        # mi faccio la media per sulle regioni per quella banda
        band_psd_mean = np.mean(band_psds, axis=0) #faccio la media per banda

        max_idx = np.argmax(band_psd_mean)
        dominant_freq = band_freqs[max_idx]
        dominant_psd_peak = band_psd_mean[max_idx]

        band_psd_dict[band] = {

            'dominant_freq': dominant_freq,
            'dominant_psd': dominant_psd_peak
        }

    return freqs, psd_all, auc_counts_all, dominant_freqs_all, band_psd_dict


def save_psds(psd_all, freqs, regions, outdir):
    #for each region, same cols
    df_psd = pd.DataFrame({"freq": freqs})

    for reg_name, psd in zip(regions, psd_all):
        df_psd[reg_name] = psd

    df_psd.to_csv(outdir+"/psd_matrix.csv", index=False)


def save_state_var(svar_all, nvar, time_sim, regions, outdir):
    #for each region, same cols
    df_svar = pd.DataFrame({"time": time_sim})

    for reg_name, svar in zip(regions, svar_all):
        df_svar[reg_name] = svar

    df_svar.to_csv(f"{outdir}/svar_matrix_{nvar}.csv.zip",index=False,compression="zip")


def save_results_to_csv(auc_counts, dominant_frequencies, regions, outdir):

    #list of rows (.append does not work in pandas 2.0)
    rows = []

    for reg_idx, (auc_data, dom_data) in enumerate(zip(auc_counts, dominant_frequencies)):
        reg_name = regions[reg_idx]

        # ogni banda è sia in auc_data che in dom_data
        for band in auc_data.keys():
            auc_values = auc_data[band]
            dom_values = dom_data[band]

            rows.append({
                'Region': reg_name,
                'Band': band,
                'AUC': auc_values['auc'],
                'SignificantCount': auc_values['significant_count'],
                'DominantFreq': dom_values['dominant_freq'],
                'DominantPSD': dom_values['dominant_psd']
            })

    results_df = pd.DataFrame(rows)

    # salvo in CSV
    results_df.to_csv(f"{outdir}/results_all.csv", index=False)


def plot_psd_with_significant_freqs(freqs, psd, bands, threshold_ratio, title_font=14, save_fig = True, plt_type='semilog'):
    max_psd = np.max(psd)
    threshold = max_psd * threshold_ratio

    significant_freqs = freqs[psd > threshold]
    significant_psd = psd[psd > threshold]

    # Colors for each band
    colors_band = {
        'delta': 'red',
        'theta': 'green',
        'alpha': 'blue',
        'beta': 'purple',
        'gamma': 'orange'
    }

    plt.figure(figsize=(12, 6))

    if plt_type=='semilog':
        plt.semilogy(significant_freqs, significant_psd, color = 'black')

    elif plt_type=='stem':
        plt.stem(significant_freqs, significant_psd, linefmt='k-', markerfmt='ko', basefmt=" ")

    elif plt_type=='db':
        plt.plot(significant_freqs, significant_psd, color = 'black')
    else:
        print('Unknown plt type - choose amongst semilog, stem or db')

    plt.xlabel('Frequency (Hz)', fontsize=title_font - 2)
    plt.ylabel('PSD', fontsize=title_font - 2)
    plt.title('Spectral analysis', fontsize=title_font)

    for band, (low, high) in bands.items():
        plt.axvspan(low, high, color=colors_band[band], alpha=0.2, label=band)

    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig('./PSD.png', dpi=300)
        print('PSD.png saved in current directory.')
    #plt.close()



def plot_average_psd_region(freq, mean_psd, psd_all, bands, colors_band, outdir, figsize=(12, 6)):

    plt.figure(figsize=figsize)
    plt.plot(freq, mean_psd, color='black', lw=2, label="PSD")
    plt.fill_between(freq,
                     np.percentile(psd_all, 25, axis=0),
                     np.percentile(psd_all, 75, axis=0),
                     color="gray", alpha=0.3, label="IQR (25-75%)")

    for band, (low, high) in bands.items():
        plt.axvspan(low, high, color=colors_band[band], alpha=0.2, label=band)

    plt.title("PSD averaged on regions", fontsize=16)
    plt.xlabel("Frequency [Hz]", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("PSD [dB]", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(outdir + '/avg_on_region_psd.pdf', dpi=300)
