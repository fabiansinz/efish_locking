from collections import defaultdict
import itertools
import pymysql
from scipy import stats
import datajoint as dj
from datajoint import schema
from schemata import Runs, GlobalEFieldPeaksTroughs, GlobalEField, SpikeTimes, peakdet, Cells
import numpy as np
from pycircstat import event_series as es
import pycircstat as circ

server = schema('efish', locals())


def compute_1st_order_spectrum(aggregated_spikes, sampling_rate, alpha=0.001):
    """
    Computes the 1st order amplitue spectrum of the spike train (i.e. the vector strength spectrum
    of the aggregated spikes).

    :param aggregated_spikes: all spike times over all trials
    :param sampling_rate: sampling rate of the spikes
    :param alpha: significance level for the boundary against non-locking
    :returns: the frequencies for the vector strength spectrum, the spectrum, and the threshold against non-locking

    """
    if len(aggregated_spikes) < 2:
        return np.array([0]), np.array([0]), 0,
    w_all, vsa_all = es.vector_strength_spectrum(aggregated_spikes, sampling_rate,
                                                 time=(np.amin(aggregated_spikes), np.amax(aggregated_spikes)))
    p = 1 - alpha
    threshold = np.sqrt(- np.log(1 - p) / len(aggregated_spikes))
    return w_all, vsa_all, threshold


def compute_2nd_order_spectrum(spikes, t, sampling_rate, alpha=0.001, method='poisson'):
    """
    Computes the 1st order amplitue spectrum of the spike train (i.e. the vector strength spectrum
    of the aggregated spikes).


    :param spikes: list of spike trains from the single trials
    :param t: numpy.array of time points
    :param sampling_rate: sampling rate of the spikes
    :param alpha: significance level for the boundary against non-locking
    :param method: method to compute the confidence interval (poisson or gauss)
    :returns: the frequencies for the vector strength spectrum, the spectrum, and the threshold against non-locking

    """

    # compute 99% confidence interval for Null distribution of 2nd order spectra (no locking)

    spikes_per_trial = list(map(len, spikes))
    freqs, vs_spectra = zip(*[es.vector_strength_spectrum(sp, sampling_rate, time=t) for sp in spikes])

    freqs = freqs[0]
    m_ampl = np.mean(vs_spectra, axis=0)

    if method == 'poisson':
        poiss_rate = np.mean(spikes_per_trial)
        r = np.linspace(0, 2, 10000)
        dr = r[1] - r[0]
        mu = np.sum(2 * poiss_rate * r ** 2 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
            1 - np.exp(-poiss_rate))) * dr
        s = np.sum(2 * poiss_rate * r ** 3 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
            1 - np.exp(-poiss_rate))) * dr
        s2 = np.sqrt(s - mu ** 2.)
        y = stats.norm.ppf(1 - alpha, loc=mu,
                           scale=s2 / np.sqrt(len(spikes_per_trial)))  # use central limit theorem

    elif method == 'gauss':
        n = np.asarray(spikes_per_trial)
        mu = np.sqrt(np.pi) / 2. * np.mean(1. / np.sqrt(n))
        N = len(spikes_per_trial)
        s = np.sqrt(np.mean(1. / n - np.pi / 4. / n) / N)
        y = stats.norm.ppf(1 - alpha, loc=mu, scale=s)
    else:
        raise ValueError("Method %s not known" % (method,))

    return freqs, m_ampl, y


def find_best_locking(spikes, fundamentals, tol=1, step=0.001):
    df = np.atleast_2d(np.arange(-tol, tol + step, step))
    max_w = []
    max_v = []
    if type(spikes) is not list:
        spikes = [spikes]

    for freq in fundamentals:
        f = freq + df
        vector_strengths = []
        for trial in spikes:
            vector_strengths.append(1 - circ.var((trial[:, None] % (1. / f)) * f * 2 * np.pi, axis=0))
        mean_vector_strength = np.mean(vector_strengths, axis=0)
        max_idx = np.argmax(mean_vector_strength)
        max_w.append(f[0, max_idx])
        max_v.append(mean_vector_strength[max_idx])

    return np.array(max_w), np.array(max_v)


def find_significant_peaks(spikes, w, spectrum, peak_dict, threshold, tol=1., upper_cutoff=2000):

    if not threshold > 0:
        print("Threshold value %.4f is not allowed" % threshold)
        return []
    # find peaks in spectrum that are greater or equal than the threshold
    max_vs, max_idx, _, _ = peakdet(spectrum, delta=threshold * .9 )
    max_vs, max_idx = max_vs[threshold <= max_vs], max_idx[threshold <= max_vs]
    max_w = w[max_idx]

    # get rid of everythings that is above the frequency cutoff
    idx = np.abs(max_w) < upper_cutoff
    if idx.sum() == 0: # no sigificant peak was found
        print('No significant peak found')
        return []
    max_w = max_w[idx]
    max_vs = max_vs[idx]



    # refine the found maxima
    max_w_ref, max_vs_ref = find_best_locking(spikes, max_w, tol=tol)

    for name, freq in peak_dict.items():
        idx = np.argmin(np.abs(max_w - freq))
        if np.abs(max_w[idx] - freq) < tol:
            print("\t\tAdjusting %s: %.2f --> %.2f" % (name, freq, max_w[idx]))
            peak_dict[name] = max_w[idx]

    coeffs = [(name, peak_dict[name], np.arange(-5, 6)) for name in peak_dict]
    coeff_names, coeff_f, coeff_facs = zip(*coeffs)

    ret = []
    for maw, ma, maw_r, ma_r in zip(max_w, max_vs, max_w_ref, max_vs_ref):
        # if maw < 0: continue
        for facs in itertools.product(*coeff_facs):
            cur_freq = np.dot(facs, coeff_f)
            if np.abs(cur_freq) > upper_cutoff:
                continue

            if np.abs(maw - cur_freq) < tol:
                tmp = dict(zip(coeff_names, facs))
                tmp['frequency'] = maw
                tmp['vector_strength'] = ma
                tmp['tolerance'] = tol
                tmp['refined'] = 0
                ret.append(tmp)

                tmp = dict(zip(coeff_names, facs))
                tmp['frequency'] = maw_r
                tmp['vector_strength'] = ma_r
                tmp['tolerance'] = tol
                tmp['refined'] = 1
                ret.append(tmp)
    return ret


@server
class FirstOrderSpikeSpectra(dj.Computed):
    definition = """
    # table that holds 1st order vector strength spectra

    -> Runs                         # each run has a spectrum

    ---

    frequencies             : longblob # frequencies at which the spectra are computed
    vector_strengths        : longblob # vector strengths at those frequencies
    critical_value          : float    # critical value for significance with alpha=0.001
    """

    @property
    def populate_relation(self):
        return Runs() & GlobalEFieldPeaksTroughs()

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        dat = (Runs() & key).fetch(as_dict=True)[0]
        dt = 1 / dat['samplingrate']

        pt = (GlobalEFieldPeaksTroughs() & key).fetch(as_dict=True)
        st = (SpikeTimes() & key).fetch(as_dict=True)

        aggregated_spikes = np.hstack([s['times'] / 1000 - p['peaks'][0] * dt for s, p in zip(st, pt)])

        key['frequencies'], key['vector_strengths'], key['critical_value'] = \
            compute_1st_order_spectrum(aggregated_spikes, 1 / dt, alpha=0.001)
        self.insert(key)


@server
class SecondOrderSpikeSpectra(dj.Computed):
    definition = """
    # table that holds 2nd order vector strength spectra
    -> Runs                  # each run has a spectrum

    ---

    frequencies             : longblob # frequencies at which the spectra are computed
    vector_strengths        : longblob # vector strengths at those frequencies
    critical_value          : float    # critical value for significance with alpha=0.001
    """

    @property
    def populate_relation(self):
        return Runs()

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        dat = (Runs() & key).fetch(as_dict=True)[0]
        dt = 1 / dat['samplingrate']
        t = np.arange(0, dat['duration'], dt)
        st = (SpikeTimes() & key).fetch(as_dict=True)
        st = [s['times'] / 1000 for s in st]  # convert to s

        key['frequencies'], key['vector_strengths'], key['critical_value'] = \
            compute_2nd_order_spectrum(st, t, 1 / dt, alpha=0.001, method='poisson')
        self.insert(key)


@server
class FirstOrderSignificantPeaks(dj.Computed):
    definition = """
    # hold significant peaks in spektra

    stimulus_coeff          : int   # how many multiples of the stimulus
    eod_coeff               : int   # how many multiples of the eod
    baseline_coeff          : int   # how many multiples of the baseline firing rate
    refined                 : int   # whether the search was refined or not
    ->FirstOrderSpikeSpectra

    ---

    frequency               : double # frequency at which there is significant locking
    vector_strength         : double # vector strength at that frequency
    tolerance               : double # tolerance within which a peak was accepted
    """

    @property
    def populate_relation(self):
        return FirstOrderSpikeSpectra()

    def _make_tuples(self, key):
        double_peaks = -1
        data = (FirstOrderSpikeSpectra() & key).fetch1()
        run = (Runs() & key).fetch1()
        cell = (Cells() & key).fetch1()

        dt = 1 / run['samplingrate']

        pt = (GlobalEFieldPeaksTroughs() & key).fetch(as_dict=True)
        st = (SpikeTimes() & key).fetch(as_dict=True)
        spikes = np.hstack([s['times'] / 1000 - p['peaks'][0] * dt for s, p in zip(st, pt)])

        interesting_frequencies = {'stimulus_coeff': run['eod'] + run['delta_f'], 'eod_coeff': run['eod'],
                                   'baseline_coeff': cell['baseline']}

        sas = find_significant_peaks(spikes, data['frequencies'], data['vector_strengths'],
                                            interesting_frequencies, data['critical_value'])
        for s in sas:
            s.update(key)
            try:
                self.insert(s)
            except pymysql.IntegrityError: # sometimes one peak has two peaks nearby
                print("Found double peak")
                s['refined'] = double_peaks
                self.insert(s)
                double_peaks -= 1


@server
class SecondOrderSignificantPeaks(dj.Computed):
    definition = """
    # hold significant peaks in spektra

    stimulus_coeff          : int   # how many multiples of the stimulus
    eod_coeff               : int   # how many multiples of the eod
    baseline_coeff          : int   # how many multiples of the baseline firing rate
    refined                 : int   # whether the search was refined or not
    ->SecondOrderSpikeSpectra

    ---

    frequency               : double # frequency at which there is significant locking
    vector_strength         : double # vector strength at that frequency
    tolerance               : double # tolerance within which a peak was accepted
    """

    @property
    def populate_relation(self):
        return SecondOrderSpikeSpectra()

    def _make_tuples(self, key):
        double_peaks = -1
        data = (SecondOrderSpikeSpectra() & key).fetch1()
        run = (Runs() & key).fetch1()
        cell = (Cells() & key).fetch1()

        dt = 1 / run['samplingrate']

        st = (SpikeTimes() & key).fetch(as_dict=True)
        spikes = [s['times'] / 1000 for s in st]  # convert to s

        interesting_frequencies = {'stimulus_coeff': run['eod'] + run['delta_f'], 'eod_coeff': run['eod'],
                                   'baseline_coeff': cell['baseline']}

        sas = find_significant_peaks(spikes, data['frequencies'], data['vector_strengths'],
                                            interesting_frequencies, data['critical_value'])
        for s in sas:
            s.update(key)
            try:
                self.insert(s)
            except pymysql.IntegrityError: # sometimes one peak has two peaks nearby
                print("Found double peak")
                s['refined'] = double_peaks
                self.insert(s)
                double_peaks -= 1


if __name__ == "__main__":
    foss = FirstOrderSpikeSpectra()
    foss.populate()

    soss = SecondOrderSpikeSpectra()
    soss.populate()

    # fosp = FirstOrderSignificantPeaks()
    # fosp.populate()

    sosp = SecondOrderSignificantPeaks()
    sosp.populate()
