import pandas as pd
import functools
import itertools
import pymysql
from scipy import stats
import datajoint as dj
from datajoint import schema
import sympy
from helpers import mkdir
from schemata import Runs, GlobalEFieldPeaksTroughs, peakdet, Cells
import numpy as np
from pycircstat import event_series as es
import pycircstat as circ
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import plot_settings
from collections import OrderedDict

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


def neg_vs_at(f, spikes):
    return -np.mean([1 - circ.var((trial % (1. / f)) * f * 2 * np.pi) for trial in spikes])


def find_best_locking(spikes, fundamentals, tol=1):
    max_w, max_v = [], []
    if type(spikes) is not list:
        spikes = [spikes]

    # at an initial and end value to fundamental to generate the search intervals
    fundamentals = np.array(fundamentals)
    fundamentals.sort()
    if fundamentals[0] > 0:
        fundamentals = np.hstack((max(fundamentals[0] - tol, 0), fundamentals, fundamentals[-1] + tol))
    elif fundamentals[-1] < 0:
        fundamentals = np.hstack((fundamentals[0] - tol, fundamentals, min(fundamentals[-1] + tol, 0)))
    else:
        fundamentals = np.hstack((fundamentals[0] - tol, fundamentals, fundamentals[-1] + tol))

    for freq_before, freq, freq_after in zip(fundamentals[:-2], fundamentals[1:-1], fundamentals[2:]):
        # search in freq +- tol unless we get too close to another fundamental. In that case, use the mid-interval
        upper = min(freq + tol, freq_after)
        lower = max(freq - tol, freq_before)
        obj = functools.partial(neg_vs_at, spikes=spikes)
        f_opt = optimize.fminbound(obj, lower, upper)
        max_w.append(f_opt)
        max_v.append(-obj(f_opt))

    return np.array(max_w), np.array(max_v)


def find_significant_peaks(spikes, w, spectrum, peak_dict, threshold, tol=2., upper_cutoff=2000):
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

    def plot(self, figbase='figures', f_max=2000, **restrictions):
        sns.set_context('paper')
        colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

        stim, eod, baseline, beat = sympy.symbols('f_s, f_e, f_b, \Delta')

        restrictions = dict(**restrictions)
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
            contrast = (Runs() & fos).fetch1['contrast']
            cell_type = (Cells() & fos).fetch1['cell_type']


            # insert refined vector strengths
            peak_f, peak_v = peaks.fetch['frequency', 'vector_strength']
            f = np.hstack((f, peak_f))
            v = np.hstack((v, peak_v))
            idx = np.argsort(f)
            f, v = f[idx], v[idx]

            # generate figure
            with sns.axes_style('ticks'):
                fig, ax = plt.subplots(figsize=(7.9, 3.93))

            # only take frequencies within defined ange
            idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            ax.fill_between(f[idx], 0 * f[idx], 0 * f[idx] + alpha, lw=0, color='silver')
            ax.fill_between(f[idx], 0 * f[idx], v[idx], lw=0, color='darkslategray')

            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('vector strength')
            ax.set_ylim((0, 1.))
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
                        term = (cs * stim + ce * eod + cb * baseline).subs(stim - eod, beat).simplify()
                        terms.append(sympy.latex(term))
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
            sns.despine(fig)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            fig.tight_layout()

            dir = figbase + '/%s' % (cell_type,)
            mkdir(dir)
            filename = dir + '/%s_%.2f%%_run%02i.pdf' % (cell, contrast, run)
            fig.savefig(filename)
            plt.close(fig)


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


if __name__ == "__main__":
    # foss = FirstOrderSpikeSpectra()
    # foss.populate(reserve_jobs=True)
    #
    # soss = SecondOrderSpikeSpectra()
    # soss.populate()

    fosp = FirstOrderSignificantPeaks()
    fosp.populate()

    sosp = SecondOrderSignificantPeaks()
    sosp.populate()
