import functools
import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
import sympy
from scipy import optimize
from scipy import stats

import datajoint as dj
import pycircstat as circ
from datajoint import schema
from djaddon import gitlog
from locking.data import Runs, GlobalEFieldPeaksTroughs, peakdet, Cells, LocalEODPeaksTroughs, Baseline
from pycircstat import event_series as es

server = schema('efish_analyses', locals())


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


def _neg_vs_at(f, spikes):
    return -np.mean([1 - circ.var((trial % (1. / f)) * f * 2 * np.pi) for trial in spikes])


def find_best_locking(spikes, f0, tol=3):
    """
    Locally searches for a maximum in vector strength for a collection of spikes.

    The vector strength is locally maximized with fminbound within f0+-tol. There are two exceptions
    to the search range:

    * if two initial guesses are closer then tol, then their mean is taken as search boundary
    * if all initial guesses are negative or positive, the search intervals are chosen such that the result is
      again negative or positive, respectively.

    :param spikes: array of spike times or list thereof
    :param f0: list of initial guesses
    :param tol: search range is +-tol in Hz
    :return: best locking frequencies, corresponding vector strength
    """
    max_w, max_v = [], []
    if type(spikes) is not list:
        spikes = [spikes]

    # at an initial and end value to fundamental to generate the search intervals
    f0 = np.array(f0)
    f0.sort()

    # --- make sure that fundamentals + boundaries stay negative positive if they were before
    if f0[0] > 0:
        f0 = np.hstack((max(f0[0] - tol, 0), f0, f0[-1] + tol))
    elif f0[-1] < 0:
        f0 = np.hstack((f0[0] - tol, f0, min(f0[-1] + tol, 0)))
    else:
        f0 = np.hstack((f0[0] - tol, f0, f0[-1] + tol))

    for freq_before, freq, freq_after in zip(f0[:-2], f0[1:-1], f0[2:]):
        # search in freq +- tol unless we get too close to another fundamental.
        upper = min(freq + tol, (freq + freq_after) / 2)
        lower = max(freq - tol, (freq_before + freq) / 2)
        obj = functools.partial(_neg_vs_at, spikes=spikes)
        f_opt = optimize.fminbound(obj, lower, upper)
        max_w.append(f_opt)
        max_v.append(-obj(f_opt))

    return np.array(max_w), np.array(max_v)


def find_significant_peaks(spikes, w, spectrum, peak_dict, threshold, tol=3.,
                           upper_cutoff=2000):  # check that search intervals work
    if not threshold > 0:
        print("Threshold value %.4f is not allowed" % threshold)
        return []
    # find peaks in spectrum that are greater or equal than the threshold
    max_vs, max_idx, _, _ = peakdet(spectrum, delta=threshold * .9)
    max_vs, max_idx = max_vs[threshold <= max_vs], max_idx[threshold <= max_vs]
    max_w = w[max_idx]

    # get rid of everythings that is above the frequency cutoff
    idx = np.abs(max_w) < upper_cutoff
    if idx.sum() == 0:  # no sigificant peak was found
        print('No significant peak found')
        return []
    max_w = max_w[idx]
    max_vs = max_vs[idx]

    # refine the found maxima
    max_w_ref, max_vs_ref = find_best_locking(spikes, max_w, tol=tol)

    # make them all sorted in the right order
    idx = np.argsort(max_w)
    max_w, max_vs = max_w[idx], max_vs[idx]
    idx = np.argsort(max_w_ref)
    max_w_ref, max_vs_ref = max_w_ref[idx], max_vs_ref[idx]

    for name, freq in peak_dict.items():
        idx = np.argmin(np.abs(max_w - freq))
        if np.abs(max_w[idx] - freq) < tol:
            print("\t\tAdjusting %s: %.2f --> %.2f" % (name, freq, max_w[idx]))
            peak_dict[name] = max_w[idx]

    coeffs = [(name, peak_dict[name], np.arange(-5, 6)) for name in peak_dict]
    coeff_names, coeff_f, coeff_facs = zip(*coeffs)

    ret = []

    for maw, ma, maw_r, ma_r in zip(max_w, max_vs, max_w_ref, max_vs_ref):
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

            if np.abs(maw_r - cur_freq) < tol:
                tmp = dict(zip(coeff_names, facs))
                tmp['frequency'] = maw_r
                tmp['vector_strength'] = ma_r
                tmp['tolerance'] = tol
                tmp['refined'] = 1
                ret.append(tmp)
    return ret


class PlotableSpectrum:
    def plot(self, ax, restrictions, f_max=2000):
        sns.set_context('paper')
        # colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
        colors = ['deeppink', 'dodgerblue', '#7570b3', 'black']

        stim, eod, baseline, beat = sympy.symbols('f_s, f_e, f_b, \Delta')

        for fos in ((self * Runs()).project() & restrictions).fetch.as_dict:
            print("Processing %(cell_id)s %(run_id)i" % fos)

            if isinstance(self, FirstOrderSpikeSpectra):
                peaks = (FirstOrderSignificantPeaks() & fos & restrictions)
            elif isinstance(self, SecondOrderSpikeSpectra):
                peaks = (SecondOrderSignificantPeaks() & fos & restrictions)
            else:
                raise Exception("Mother class unknown!")

            f, v, alpha, cell, run = (self & fos).fetch1['frequencies', 'vector_strengths', 'critical_value',
                                                         'cell_id', 'run_id']

            # insert refined vector strengths
            peak_f, peak_v = peaks.fetch['frequency', 'vector_strength']
            f = np.hstack((f, peak_f))
            v = np.hstack((v, peak_v))
            idx = np.argsort(f)
            f, v = f[idx], v[idx]

            # only take frequencies within defined ange
            idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            ax.fill_between(f[idx], 0 * f[idx], 0 * f[idx] + alpha, lw=0, color='silver')
            ax.fill_between(f[idx], 0 * f[idx], v[idx], lw=0, color='darkslategray')

            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('vector strength')
            ax.set_ylim((0, 1.3))
            ax.set_xlim((0, f_max))
            ax.set_yticks([0, .25, .5, .75, 1.0])

            df = pd.DataFrame(peaks.fetch())
            df['on'] = np.abs(df.ix[:, :3]).sum(axis=1)
            df = df[df.frequency > 0]

            for freq, freq_group in df.groupby('frequency'):  # get all combinations that have the same frequency

                freq_group = freq_group[
                    freq_group.on == freq_group.on.min()]  # take the ones that have the lowest factors

                for i, (cs, ce, cb, _, _, _, _, freq, vs, _, _) in freq_group.iterrows():
                    terms = []
                    if 0 <= freq <= f_max:
                        term = cs * stim + ce * eod + cb * baseline
                        if (cs < 0 and ce > 0) or (cs > 0 and ce < 0):
                            coeff = np.sign(ce) * min(abs(cs), abs(ce))
                            term = term + coeff * (stim - eod) - coeff * beat
                        terms.append(sympy.latex(term.simplify()))
                    term = ' = '.join(terms)

                    # use different colors and labels depending on the frequency
                    if cs != 0 and ce == 0 and cb == 0:
                        ax.plot(freq, vs, 'ok', mfc=colors[0], label='stimulus')
                    elif cs == 0 and ce != 0 and cb == 0:
                        ax.plot(freq, vs, 'ok', mfc=colors[1], label='EOD')
                    elif cs == 0 and ce == 0 and cb != 0:
                        ax.plot(freq, vs, 'ok', mfc=colors[2], label='baseline firing')
                    else:
                        ax.plot(freq, vs, 'ok', mfc=colors[3], label='combinations')
                    ax.text(freq, vs + .05, r'$%s=%.1f$Hz' % (term, freq), fontsize=5, rotation=80,
                            ha='left',
                            va='bottom')
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())


@server
class FirstOrderSpikeSpectra(dj.Computed, PlotableSpectrum):
    definition = """
    # table that holds 1st order vector strength spectra

    -> Runs                         # each run has a spectrum

    ---

    frequencies             : longblob # frequencies at which the spectra are computed
    vector_strengths        : longblob # vector strengths at those frequencies
    critical_value          : float    # critical value for significance with alpha=0.001
    """

    @property
    def populated_from(self):
        return Runs() & GlobalEFieldPeaksTroughs()

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        dt = 1. / (Runs() & key).fetch1['samplingrate']

        trials = ((GlobalEFieldPeaksTroughs() * Runs.SpikeTimes()) & key)

        aggregated_spikes = np.hstack([s / 1000 - p[0] * dt for s, p in zip(*trials.fetch['times', 'peaks'])])
        # aggregated_spikes = np.hstack([s['times'] / 1000 - p['peaks'][0] * dt for s, p in zip(st, pt)])

        key['frequencies'], key['vector_strengths'], key['critical_value'] = \
            compute_1st_order_spectrum(aggregated_spikes, 1 / dt, alpha=0.001)
        vs = key['vector_strengths']
        vs[np.isnan(vs)] = 0
        self.insert1(key)

@server
@gitlog
class StimulusSpikeJitter(dj.Computed):
    definition = """
    # circular variance and std of spike times within an EOD period during stimulation

    -> Runs

    ---
    stim_var         : double # circular variance
    stim_std         : double # circular std
    stim_mean        : double # circular mean
    """

    @property
    def populated_from(self):
        return Runs() & LocalEODPeaksTroughs()

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        dt = 1. / (Runs() & key).fetch1['samplingrate']
        eod = (Runs() & key).fetch1['eod']
        trials = ((LocalEODPeaksTroughs() * Runs.SpikeTimes()) & key)

        aggregated_spikes = np.hstack([s / 1000 - p[0] * dt for s, p in zip(*trials.fetch['times', 'peaks'])])

        aggregated_spikes %= 1 / eod

        aggregated_spikes *= eod * 2 * np.pi  # normalize to 2*pi
        key['stim_var'], key['stim_mean'], key['stim_std'] = \
            circ.var(aggregated_spikes), circ.mean(aggregated_spikes), circ.std(aggregated_spikes)
        self.insert1(key)


@server
@gitlog
class BaselineSpikeJitter(dj.Computed):
    definition = """
    # circular variance and mean of spike times within an EOD period

    -> Baseline

    ---

    base_var         : double # circular variance
    base_std         : double # circular std
    base_mean        : double # circular mean
    """

    @property
    def populated_from(self):
        return Baseline() & Baseline.LocalEODPeaksTroughs() & dict(cell_type='p-unit')

    def _make_tuples(self, key):
        print('Processing', key['cell_id'])
        sampling_rate, eod = (Baseline() & key).fetch1['samplingrate', 'eod']
        dt = 1. / sampling_rate

        trials = Baseline.LocalEODPeaksTroughs() * Baseline.SpikeTimes() & key

        aggregated_spikes = np.hstack([s / 1000 - p[0] * dt for s, p in zip(*trials.fetch['times', 'peaks'])])

        aggregated_spikes %= 1 / eod

        aggregated_spikes *= eod * 2 * np.pi  # normalize to 2*pi
        key['base_var'], key['base_mean'], key['base_std'] = \
            circ.var(aggregated_spikes), circ.mean(aggregated_spikes), circ.std(aggregated_spikes)
        self.insert1(key)


@server
class SecondOrderSpikeSpectra(dj.Computed, PlotableSpectrum):
    definition = """
    # table that holds 2nd order vector strength spectra
    -> Runs                  # each run has a spectrum

    ---

    frequencies             : longblob # frequencies at which the spectra are computed
    vector_strengths        : longblob # vector strengths at those frequencies
    critical_value          : float    # critical value for significance with alpha=0.001
    """

    def _make_tuples(self, key):
        print('Processing', key['cell_id'], 'run', key['run_id'], )
        dat = (Runs() & key).fetch(as_dict=True)[0]
        dt = 1 / dat['samplingrate']
        t = np.arange(0, dat['duration'], dt)
        st = (Runs.SpikeTimes() & key).fetch(as_dict=True)
        st = [s['times'] / 1000 for s in st]  # convert to s

        key['frequencies'], key['vector_strengths'], key['critical_value'] = \
            compute_2nd_order_spectrum(st, t, 1 / dt, alpha=0.001, method='poisson')
        self.insert1(key)


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

    def _make_tuples(self, key):
        double_peaks = -1
        data = (FirstOrderSpikeSpectra() & key).fetch1()
        run = (Runs() & key).fetch1()
        cell = (Cells() & key).fetch1()

        dt = 1 / run['samplingrate']
        trials = ((GlobalEFieldPeaksTroughs() * Runs.SpikeTimes()) & key)

        # pt = (GlobalEFieldPeaksTroughs() & key).fetch(as_dict=True)
        # st = (Runs.SpikeTimes() & key).fetch(as_dict=True)
        # spikes = np.hstack([s['times'] / 1000 - p['peaks'][0] * dt for s, p in zip(st, pt)])
        spikes = np.hstack([s / 1000 - p[0] * dt for s, p in zip(*trials.fetch['times', 'peaks'])])

        interesting_frequencies = {'stimulus_coeff': run['eod'] + run['delta_f'], 'eod_coeff': run['eod'],
                                   'baseline_coeff': cell['baseline']}

        sas = find_significant_peaks(spikes, data['frequencies'], data['vector_strengths'],
                                     interesting_frequencies, data['critical_value'])
        for s in sas:
            s.update(key)
            try:
                self.insert1(s)
            except pymysql.IntegrityError:  # sometimes one peak has two peaks nearby
                print("Found double peak")
                s['refined'] = double_peaks
                self.insert1(s)
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

    def _make_tuples(self, key):
        double_peaks = -1
        data = (SecondOrderSpikeSpectra() & key).fetch1()
        run = (Runs() & key).fetch1()
        cell = (Cells() & key).fetch1()

        dt = 1 / run['samplingrate']

        st = (Runs.SpikeTimes() & key).fetch(as_dict=True)
        spikes = [s['times'] / 1000 for s in st]  # convert to s

        interesting_frequencies = {'stimulus_coeff': run['eod'] + run['delta_f'], 'eod_coeff': run['eod'],
                                   'baseline_coeff': cell['baseline']}

        sas = find_significant_peaks(spikes, data['frequencies'], data['vector_strengths'],
                                     interesting_frequencies, data['critical_value'])
        for s in sas:
            s.update(key)

            try:
                self.insert1(s)
            except pymysql.IntegrityError:  # sometimes one peak has two peaks nearby
                print("Found double peak")
                s['refined'] = double_peaks
                self.insert1(s)
                double_peaks -= 1


@server
class SamplingPointsPerBin(dj.Lookup):
    definition = """
    # sampling points per bin

    n           : int # sampling points per bin
    ---

    """

    contents = [(2,), (4,), (8,)]


@server
class PhaseLockingHistogram(dj.Computed):
    definition = """
    # phase locking histogram at significant peaks

    -> FirstOrderSignificantPeaks
    ---
    locking_frequency       : double   # frequency for which the locking is computed
    peak_frequency          : double   # frequency as determined by the peaks of the electric field
    spikes                  : longblob # union of spike times over trials relative to period of locking frequency
    vector_strength         : double   # vector strength computed from the spikes for sanity checking
    """

    class Histograms(dj.Part):
        definition = """
        ->PhaseLockingHistogram
        ->SamplingPointsPerBin
        ---
        bin_width_radians       : double   # bin width in radians
        bin_width_time          : double   # bin width in time
        histogram               : longblob # vector of counts
        """

    @property
    def populated_from(self):
        return FirstOrderSignificantPeaks() \
               & 'baseline_coeff=0' \
               & '((stimulus_coeff=1 and eod_coeff=0) or (stimulus_coeff=0 and eod_coeff=1))' \
               & 'refined=1'

    def _make_tuples(self, key):
        key_sub = dict(key)
        delta_f, eod, samplingrate = (Runs() & key).fetch1['delta_f', 'eod', 'samplingrate']
        locking_frequency = (FirstOrderSignificantPeaks() & key).fetch1['frequency']

        if key['eod_coeff'] > 0:
            # convert spikes to s and center on first peak of eod
            times, peaks = (Runs.SpikeTimes() * LocalEODPeaksTroughs() & key).fetch['times', 'peaks']

            spikes = np.hstack([s / 1000 - p[0] / samplingrate for s, p in zip(times, peaks)])
        else:
            # convert spikes to s and center on first peak of stimulus
            times, peaks = (Runs.SpikeTimes() * GlobalEFieldPeaksTroughs() & key).fetch['times', 'peaks']
            spikes = np.hstack([s / 1000 - p[0] / samplingrate for s, p in zip(times, peaks)])

        key['peak_frequency'] = samplingrate / np.mean([np.diff(p).mean() for p in peaks])
        key['locking_frequency'] = locking_frequency

        cycle = 1 / locking_frequency
        spikes %= cycle

        key['spikes'] = spikes / cycle * 2 * np.pi
        key['vector_strength'] = 1 - circ.var(key['spikes'])

        self.insert1(key)

        histograms = self.Histograms()
        for n in SamplingPointsPerBin().fetch:
            n = int(n[0])
            bin_width_time = n / samplingrate
            bin_width_radians = bin_width_time / cycle * np.pi * 2
            bins = np.arange(0, cycle + bin_width_time, bin_width_time)
            key_sub['n'] = n
            key_sub['histogram'], _ = np.histogram(spikes, bins=bins)
            key_sub['bin_width_time'] = bin_width_time
            key_sub['bin_width_radians'] = bin_width_radians

            histograms.insert1(key_sub)

    def violin_plot(self, ax, restrictions, palette):
        runs = Runs() * self & restrictions
        if len(runs) == 0:
            return

        df = pd.concat([pd.DataFrame(item) for item in runs.fetch.as_dict()])
        df.ix[df.stimulus_coeff == 1, 'type'] = 'stimulus'
        df.ix[df.eod_coeff == 1, 'type'] = 'EOD'
        delta_fs = np.unique(runs.fetch['delta_f'])
        delta_fs = delta_fs[np.argsort(-delta_fs)]

        sns.violinplot(data=df, y='delta_f', x='spikes', hue='type', split=True, ax=ax, hue_order=['EOD', 'stimulus'],
                       order=delta_fs, palette=palette, cut=0, inner=None, linewidth=.5,
                       orient='h', bw=.05)


@server
class CoincidenceTolerance(dj.Lookup):
    definition = """
    # Coincidence tolerance of EOD and stimulus phase in s

    coincidence_idx         : int
    ---
    tol                     : double
    """

    contents = [(0, 0.0001), ]


@server
class EODStimulusPSTSpikes(dj.Computed):
    definition = """
    # PSTH of Stimulus and EOD at the difference frequency of both

    -> FirstOrderSignificantPeaks
    -> CoincidenceTolerance
    cycle_idx                : int  # index of the cycle
    ---
    stimulus_frequency       : double
    eod_frequency            : double
    window_half_size         : double # spikes will be extracted around +- this size around in phase points of stimulus and eod
    vector_strength_eod      : double
    vector_strength_stimulus : double
    spikes                   : longblob
    """

    @property
    def populated_from(self):
        constr = dict(stimulus_coeff=1, baseline_coeff=0, eod_coeff=0, refined=1)
        cell_type = Cells() & dict(cell_type='p-unit')
        return FirstOrderSignificantPeaks() & cell_type & constr

    def _make_tuples(self, key):
        # key_sub = dict(key)
        delta_f, eod, samplingrate = (Runs() & key).fetch1['delta_f', 'eod', 'samplingrate']
        runs_stim = Runs() * FirstOrderSignificantPeaks() & key
        runs_eod = Runs() * FirstOrderSignificantPeaks() & dict(key, stimulus_coeff=0, eod_coeff=1)

        if len(runs_eod) > 0:
            # duration = runs_eod.fetch1['duration']
            tol = (CoincidenceTolerance() & key).fetch1['tol']
            eod_period = 1 / runs_eod.fetch1['frequency']

            whs = 10 * eod_period
            times, peaks, epeaks = (Runs.SpikeTimes() * LocalEODPeaksTroughs() \
                                    * GlobalEFieldPeaksTroughs().project(epeaks='peaks') \
                                    & key).fetch['times', 'peaks', 'epeaks']

            p0 = [peaks[i][
                      np.abs(epeaks[i][:, None] - peaks[i][None, :]).min(axis=0) <= tol * samplingrate] / samplingrate
                  for i in range(len(peaks))]

            spikes = []

            for train, in_phase in zip(times, p0):
                train /= 1000  # convert to seconds
                for phase in in_phase:
                    chunk = train[(train >= phase - whs) & (train <= phase + whs)] - phase
                    if len(chunk) > 0:
                        spikes.append(chunk)

            key['eod_frequency'] = runs_eod.fetch1['frequency']
            key['vector_strength_eod'] = runs_eod.fetch1['vector_strength']
            key['stimulus_frequency'] = runs_stim.fetch1['frequency']
            key['vector_strength_stimulus'] = runs_stim.fetch1['vector_strength']
            key['window_half_size'] = whs

            for cycle_idx, train in enumerate(spikes):
                key['spikes'] = train
                key['cycle_idx'] = cycle_idx
                self.insert1(key)

    def plot(self, ax, restrictions, coincidence=0.0001):
        rel = self * CoincidenceTolerance() * Runs().project('delta_f') & restrictions & dict(tol=coincidence)
        df = pd.DataFrame(rel.fetch())
        df['adelta_f'] = np.abs(df.delta_f)
        df['sdelta_f'] = np.sign(df.delta_f)
        df.sort(['adelta_f', 'sdelta_f'], inplace=True)

        if len(df) > 0:
            whs = df.window_half_size.mean()
            # eod = df.eod_frequency.mean()
            y = []
            yticks = []
            old = np.Inf
            for i, (sp, del_f) in enumerate(zip(df.spikes, df.delta_f)):
                if del_f != old:
                    y.append(i)
                    old = del_f
                    yticks.append(del_f)
                ax.plot(sp, 0 * sp + i, '.k', mfc='k', ms=.5, zorder=-10, rasterized=False)
            y.append(i)
            y = np.asarray(y)

            ax.set_xlim((-whs, whs))
            ax.set_xticks([-whs, -whs / 2, 0, whs / 2, whs])

            # ax.set_xticklabels(['-10 EOD', '0', '10 EOD'])
            ax.set_xticklabels([-10, -5, 0, 5, 10])
            # ax.set_xlabel('EOD cycles')
            ax.set_ylabel(r'$\Delta f$')
            ax.tick_params(axis='y', length=0, width=0, which='major')

            for y_from, y_to in zip(y[::2], y[1::2]):
                ax.fill_between([-whs, whs], [y_from, y_from], [y_to, y_to], color='gainsboro', zorder=-20)
            ax.set_yticks(0.5 * (y[1:] + y[:-1]))
            ax.set_yticklabels(yticks)
            ax.set_ylim(y[[0, -1]])


if __name__ == "__main__":
    # time.sleep(np.random.rand() * 10)
    # foss = FirstOrderSpikeSpectra()
    # foss.populate(reserve_jobs=True)
    #
    # soss = SecondOrderSpikeSpectra()
    # soss.populate(reserve_jobs=True)
    #
    # fosp = FirstOrderSignificantPeaks()
    # fosp.populate(reserve_jobs=True)
    #
    # sosp = SecondOrderSignificantPeaks()
    # sosp.populate(reserve_jobs=True)
    #
    # plh = PhaseLockingHistogram()
    # plh.populate(reserve_jobs=True)

    # EODStimulusPSTSpikes().populate(reserve_jobs=True)
    BaselineSpikeJitter().populate(reserve_jobs=True)
