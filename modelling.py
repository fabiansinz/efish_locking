import pandas as pd
import datajoint as dj
from datajoint import schema
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from djaddon import gitlog
from schemata import EFishes, peakdet, Runs
import warnings
from scipy.signal import butter, filtfilt

schema = schema('efish_locking', locals())


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


#
# def detect_peaks_and_troughs(normalized_eod, peak_threshold=None):
#     if peak_threshold is None:
#         peak_threshold = np.percentile(np.abs(normalized_eod), 99.9) - np.percentile(np.abs(normalized_eod), 50)
#         _, eod_peak_idx, _, eod_trough_idx = peakdet(normalized_eod, peak_threshold,
#                                                      np.arange(len(normalized_eod), dtype=int))
#     return normalized_eod[eod_peak_idx], eod_peak_idx, normalized_eod[eod_trough_idx], eod_trough_idx


def normalize_signal(eod, samplerate, norm_window=.5):
    max_time = len(eod) / samplerate

    if norm_window > max_time * .5:
        warnings.warn("norm_window is larger than trace. Not normalizing anything!")
        return eod

    w = np.ones(samplerate * norm_window)
    w[:] /= len(w)
    local_std = np.sqrt(np.correlate(eod ** 2., w, mode='same') - np.correlate(eod, w, mode='same') ** 2.)
    local_mean = np.correlate(eod, w, mode='same')
    return (eod - local_mean) / local_std


def amplitude_spec(dat, samplerate):
    return np.abs(np.fft.fft(dat)), np.fft.fftfreq(len(dat), 1. / samplerate)


def get_fundamental_frequency_estimates(filtered_data, samplerate, four_search_range=(-20, 20)):
    n = len(filtered_data)
    t = np.arange(n) / samplerate

    eod_peaks1, eod_peak_idx1, eod_troughs1, eod_troughs_idx1 = detect_peaks_and_troughs(filtered_data)

    diff_eod_peak_t = np.diff(t[eod_peak_idx1])

    median_period_length = np.median(diff_eod_peak_t)
    freq_from_median = 1 / median_period_length
    mean_period_length = np.mean(diff_eod_peak_t)
    freq_from_mean = 1 / mean_period_length

    f, w = amplitude_spec(filtered_data, samplerate)

    four2 = np.array(f)
    four2[(w < freq_from_median + four_search_range[0]) & (w > freq_from_median + four_search_range[1])] = -np.Inf
    freq_from_fourier = np.argmax(four2)

    return abs(w[freq_from_fourier]), freq_from_median, freq_from_mean


def estimate_fundamental(dat, samplerate, highcut=3000, normalize=-1, four_search_range=(-20, 20)):
    """
    Estimates the fundamental frequency in the data.

    :param dat: one dimensional array
    :param samplerate: sampling rate of that array
    :param highcut: highcut for the filter
    :param normalize: whether to normalize the data or not
    :param four_search_range: search range in the Fourier domain in Hz
    :return: fundamental frequency
    """
    filtered_data = butter_lowpass_filter(dat, highcut, samplerate, order=5)

    if normalize > 0:
        filtered_data = normalize_signal(filtered_data, samplerate, norm_window=normalize)

    n = len(filtered_data)
    t = np.arange(n) / samplerate

    _, eod_peak_idx, _, eod_trough_idx = peakdet(filtered_data)

    diff_eod_peak_t = np.diff(t[eod_peak_idx])
    freq_from_median = 1 / np.median(diff_eod_peak_t)
    f, w = amplitude_spec(filtered_data, samplerate)

    f[(w < freq_from_median + four_search_range[0]) & (w > freq_from_median + four_search_range[1])] = -np.Inf
    freq_from_fourier = np.argmax(f)

    return abs(w[freq_from_fourier])


def get_best_time_window(data, samplerate, fundamental_frequency, eod_cycles):
    eod_peaks1, eod_peak_idx1, _, _ = peakdet(data)

    max_time = len(data) / samplerate
    time_for_eod_cycles_in_window = eod_cycles / fundamental_frequency

    if time_for_eod_cycles_in_window > max_time * .2:
        time_for_eod_cycles_in_window = max_time * .2
        warnings.warn("You are reqeusting a window that is too long. Using T=%f" % (time_for_eod_cycles_in_window,))

    sample_points_in_window = int(fundamental_frequency * time_for_eod_cycles_in_window)

    tApp = np.arange(len(data)) / samplerate
    w1 = np.ones(sample_points_in_window) / sample_points_in_window

    local_mean = np.correlate(eod_peaks1, w1, mode='valid')
    local_std = np.sqrt(np.correlate(eod_peaks1 ** 2., w1, mode='valid') - local_mean ** 2.)
    COV = local_std / local_mean

    mi = min(COV)
    for ind, j in enumerate(COV):
        if j == mi:
            v = (eod_peak_idx1[ind])

    idx = (tApp >= tApp[v]) & (tApp < tApp[v] + time_for_eod_cycles_in_window)
    tApp = tApp[idx]
    dat_app = data[idx]
    tApp = tApp - tApp[0]

    return tApp, dat_app


def get_harm_coeff(time, dat, fundamental_freq, harmonics):
    ret = np.zeros((harmonics, 2))
    VR = fundamental_freq * 2. * np.pi
    # combCoeff = np.zeros((harmonics, 1))

    rec = 0 * time

    for i, ti in enumerate(np.arange(1, harmonics + 1)):
        V1 = np.sin(time * ti * VR)
        V2 = np.cos(time * ti * VR)
        V1 = V1 / np.sqrt(sum(V1 ** 2.))
        V2 = V2 / np.sqrt(sum(V2 ** 2.))

        coeff_sin, coeff_cos = np.dot(V1, dat), np.dot(V2, dat)

        VS = coeff_sin * V1
        VC = coeff_cos * V2
        rec = rec + VS + VC

        ret[i, :] = [coeff_sin, coeff_cos]

    return ret  # combCoeff


# =======================================================================================

@schema
class NoHarmonics(dj.Lookup):
    definition = """
    no_harmonics        : int  # number of harmonics that are fitted
    """

    contents = [(8,)]


@schema
@gitlog
class EODFit(dj.Computed):
    definition = """
    ->EFishes
    ->NoHarmonics
    ---
    fundamental     : double    # fundamental frequency
    """

    def _make_tuples(self, key):
        dat = (Runs() * Runs.LocalEOD() & key).fetch.limit(1).as_dict()[0]  # get some EOD trace
        w0 = estimate_fundamental(dat['local_efield'], dat['samplingrate'], highcut=3000, normalize=.5)
        t, win = get_best_time_window(dat['local_efield'], dat['samplingrate'], w0, eod_cycles=10)

        fundamental = estimate_fundamental(win, dat['samplingrate'], highcut=3000)
        assert abs(fundamental - dat['eod']) < 1, \
            "EOD and fundamental estimation are more than 1Hz apart: %.2fHz, %.2fHz" % (fundamental, dat['eod'])
        harm_coeff = get_harm_coeff(t, win, fundamental, key['no_harmonics'])

        self.insert1(dict(key, fundamental=fundamental))
        for key['harmonic'], (key['sin'], key['cos']) in enumerate(harm_coeff):
            EODFit.Harmonic().insert1(key)

    class Harmonic(dj.Part):
        definition = """
        # sin and cos coefficient for harmonics
        -> EODFit
        harmonic        : int   # 0th is fundamental, 1st is frist harmonic and so on
        ---
        sin             : double # coefficient for sin
        cos             : double # coefficient for cos
        """

    def generate_eod(self, t, key):
        w = (self & key).fetch1['fundamental']

        ret = 0 * t
        VR = w * 2. * np.pi
        for i, coeff_sin, coeff_cos in zip(*(self.Harmonic() & key).fetch['harmonic', 'sin', 'cos']):
            V1 = np.sin(t * (i + 1) * VR)
            V2 = np.cos(t * (i + 1) * VR)
            V1 = V1 / np.sqrt(sum(V1 ** 2.))
            V2 = V2 / np.sqrt(sum(V2 ** 2.))

            VS = coeff_sin * V1
            VC = coeff_cos * V2
            ret = ret + VS + VC
        return ret

    def eod_func(self, key):
        harm_coeff = np.vstack((EODFit.Harmonic() & key).fetch.order_by('harmonic')['sin', 'cos']).T
        fundamental = (self & key).fetch1['fundamental']

        A = np.sqrt(np.sum(harm_coeff[0, :] ** 2))

        def ret_func(t):
            vr = fundamental * 2 * np.pi
            ret = 0 * t
            for i, (coeff_sin, coeff_cos) in enumerate(harm_coeff / A):
                ret += coeff_sin * np.sin(t * (i + 1) * vr)
                ret += coeff_cos * np.cos(t * (i + 1) * vr)
            return ret

        return ret_func

@schema
class LIFPUnit(dj.Lookup):
    definition = """
    # parameters for a LIF P-Unit simulation

    id           : varchar(100) # non-double unique identifier
    ---
    zeta            : double
    eod             : double    # EOD frequency
    resonant_freq   : double    # resonant frequency of the osciallator
    tau             : double
    gain            : double
    offset          : double
    noise_sd        : double
    threshold       : double
    reset           : double
    lif_tau         : double
    """

    contents = [
        # TODO insert data from NWG poster here
    ]
if __name__ == '__main__':
    EODFit().populate(reserve_jobs=True)
