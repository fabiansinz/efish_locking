import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.integrate import odeint
from scipy.signal import butter, filtfilt

import datajoint as dj
import pycircstat as circ
import datajoint as dj
from djaddon import gitlog
from locking.data import peakdet, Runs, Cells, LocalEODPeaksTroughs, CenteredPUnitPhases, UncenteredPUnitPhases, EFishes

schema = dj.schema('efish_modelling', locals())


def second_order_critical_vector_strength(spikes, alpha=0.001):
    spikes_per_trial = [len(s) for s in spikes]
    poiss_rate = np.mean(spikes_per_trial)
    r = np.linspace(0, 2, 10000)
    dr = r[1] - r[0]
    mu = np.sum(2 * poiss_rate * r ** 2 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
        1 - np.exp(-poiss_rate))) * dr
    s = np.sum(2 * poiss_rate * r ** 3 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
        1 - np.exp(-poiss_rate))) * dr
    s2 = np.sqrt(s - mu ** 2.)
    threshold = stats.norm.ppf(1 - alpha, loc=mu,
                               scale=s2 / np.sqrt(len(spikes_per_trial)))  # use central limit theorem

    return threshold


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


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
@gitlog
class LIFPUnit(dj.Computed):
    definition = """
    # parameters for a LIF P-Unit simulation

    id           : varchar(100) # non-double unique identifier
    ->EODFit
    ---
    zeta            : double
    resonant_freq   : double    # resonant frequency of the osciallator in Hz
    tau             : double
    gain            : double
    offset          : double
    noise_sd        : double
    threshold       : double
    reset           : double
    lif_tau         : double
    """

    def _make_tuples(self, key):
        eod = (EODFit() & key).fetch1['fundamental']
        self.insert1(dict(key, id='nwgimproved',
                          zeta=0.2,
                          tau=0.002,
                          resonant_freq=eod,
                          gain=70,
                          offset=9,
                          noise_sd=30,
                          threshold=14.,
                          reset=0.,
                          lif_tau=0.001
                          ))

    def simulate(self, key, n, t, stimulus, y0=None):
        """
        Samples spikes from leaky integrate and fire neuron with id==settings_name and time t.
        Returns n trials

        :param key: key that uniquely identifies a setting
        :param n: number of trials
        :param t: time array
        :param stimulus: stimulus as a function of time (function handle)
        :return: spike times
        """

        # --- get parameters from database
        zeta, tau, gain, wr, lif_tau, offset, threshold, reset, noisesd = (self & key).fetch1[
            'zeta', 'tau', 'gain', 'resonant_freq', 'lif_tau', 'offset', 'threshold', 'reset', 'noise_sd']
        wr *= 2 * np.pi
        w0 = wr / np.sqrt(1 - 2 * zeta ** 2)
        Zm = np.sqrt((2 * w0 * zeta) ** 2 + (wr ** 2 - w0 ** 2) ** 2 / wr ** 2)
        alpha = wr * Zm

        # --- set initial values if not given
        if y0 is None:
            y0 = np.zeros(3)

        # --- differential equations for resonantor
        def _d(y, t):
            return np.array([
                y[1],
                stimulus(t) - 2 * zeta * w0 * y[1] - w0 ** 2 * y[0],
                (-y[2] + gain * alpha * max(y[0], 0)) / tau
            ])

        # --- simulate LIF
        dt = t[1] - t[0]

        Vin = odeint(lambda y, tt: _d(y, tt), y0, t).T[2]
        Vin -= offset

        Vout = np.zeros(n)

        ret = [list() for _ in range(n)]

        sdB = np.sqrt(dt) * noisesd

        for i, T in enumerate(t):
            Vout += (-Vout + Vin[i]) * dt / lif_tau + np.random.randn(n) * sdB
            idx = Vout > threshold
            for j in np.where(idx)[0]:
                ret[j].append(T)
            Vout[idx] = reset

        return tuple(np.asarray(e) for e in ret), Vin


@schema
@gitlog
class PUnitSimulations(dj.Computed):
    definition = """
    # LIF simulations

    ->LIFPUnit
    ->Runs
    ---
    dt              : double # time resolution for differential equation
    duration        : double # duration of trial in seconds
    """

    @property
    def populated_from(self):
        return (LIFPUnit() * Runs() * Cells() & dict(am=0, n_harmonics=0, cell_type='p-unit', contrast=20)).project()

    def _make_tuples(self, key):
        dt, duration = 0.000005, 1
        trials = 50
        ikey = dict(key)
        ikey['dt'] = dt
        ikey['duration'] = duration

        eod = (EODFit() & key).fetch1['fundamental']

        delta_f = (Runs() & key).fetch1['delta_f']
        other_eod = eod + delta_f

        t = np.arange(0, duration, dt)

        baseline = EODFit().eod_func(key)
        bl = baseline(t)
        fac = (bl.max() - bl.min()) * 0.2 / 2
        stimulus = lambda tt: baseline(tt) + fac * np.sin(2 * np.pi * other_eod * tt)

        spikes_base, membran_base = LIFPUnit().simulate(key, trials, t, baseline)
        spikes_stim, membran_stim = LIFPUnit().simulate(key, trials, t, stimulus)

        n = int(duration / dt)
        w = np.fft.fftfreq(n, d=dt)
        w = w[(w >= 0) & (w <= 3000)]
        vs = np.mean([circ.event_series.direct_vector_strength_spectrum(sp, w) for sp in spikes_stim], axis=0)
        ci = second_order_critical_vector_strength(spikes_stim)

        self.insert1(ikey)

        for i, (bsp, ssp) in enumerate(zip(spikes_base, spikes_stim)):
            PUnitSimulations.BaselineSpikes().insert1(dict(key, trial_idx=i, times=bsp))
            PUnitSimulations.StimulusSpikes().insert1(dict(key, trial_idx=i, times=ssp))

        PUnitSimulations.BaselineMembranePotential().insert1(dict(key, potential=membran_base))
        PUnitSimulations.StimulusMembranePotential().insert1(dict(key, potential=membran_stim))
        PUnitSimulations.Baseline().insert1(dict(key, signal=bl))
        PUnitSimulations.Stimulus().insert1(dict(key, signal=stimulus(t)))
        PUnitSimulations.StimulusSecondOrderSpectrum().insert1(dict(key, spectrum=vs, ci=ci, freq=w))

    class BaselineSpikes(dj.Part):
        definition = """
        # holds the simulated spiketimes

        ->PUnitSimulations
        trial_idx       : int # index of trial
        ---
        times           : longblob # spike times
        """

    class StimulusSpikes(dj.Part):
        definition = """
        # holds the simulated spiketimes

        ->PUnitSimulations
        trial_idx       : int # index of trial
        ---
        times           : longblob # spike times
        """

    class StimulusSecondOrderSpectrum(dj.Part):
        definition = """
        # holds the vector strength spectrum of simulated spiketimes

        ->PUnitSimulations
        ---
        freq               : longblob # frequencies at which the vector strengths are computed
        spectrum           : longblob # spike times
        ci                 : double   # (1-0.001) confidence interval
        """

    class BaselineMembranePotential(dj.Part):
        definition = """
        # holds the simulated membrane potential

        ->PUnitSimulations
        ---
        potential       : longblob # membrane potential
        """

    class StimulusMembranePotential(dj.Part):
        definition = """
        # holds the simulated membrane potential

        ->PUnitSimulations
        ---
        potential       : longblob # membrane potential
        """

    class Baseline(dj.Part):
        definition = """
        # holds the simulated membrane potential

        ->PUnitSimulations
        ---
        signal       : longblob # membrane potential
        """

    class Stimulus(dj.Part):
        definition = """
        # holds the simulated membrane potential

        ->PUnitSimulations
        ---
        signal       : longblob # membrane potential
        """

    def plot_stimulus_spectrum(self, key, ax, f_max=2000):
        dt = (self & key).fetch1['dt']
        eod = (EODFit() & key).fetch1['fundamental']
        eod2 = eod + (Runs() & key).fetch1['delta_f']

        stimulus_signal = (PUnitSimulations.Stimulus() & key).fetch1['signal']
        w = np.fft.fftfreq(len(stimulus_signal), d=dt)
        idx = (w > 0) & (w < f_max)

        S = np.abs(np.fft.fft(stimulus_signal))
        S /= S.max()

        ax.fill_between(w[idx], 0 * w[idx], S[idx], color='dodgerblue')

        # --- get parameters from database
        zeta, tau, gain, wr, lif_tau, offset, threshold, reset, noisesd = (LIFPUnit() & key).fetch1[
            'zeta', 'tau', 'gain', 'resonant_freq', 'lif_tau', 'offset', 'threshold', 'reset', 'noise_sd']
        wr *= 2 * np.pi
        w0 = wr / np.sqrt(1 - 2 * zeta ** 2)

        w2 = w * 2 * np.pi
        Zm = np.sqrt((2 * w0 * zeta) ** 2 + (w2 ** 2 - w0 ** 2) ** 2 / w2 ** 2)
        dho = 1. / (w2[idx] * Zm[idx])

        ax.plot(w[idx], dho / np.nanmax(dho), '--', dashes=(2, 2), color='gray', label='harmonic oscillator', lw=1,
                zorder=-10)
        lp = 1. / np.sqrt(w2[idx] ** 2 * tau ** 2 + 1)
        ax.plot(w[idx], lp / lp.max(), '--', color='gray', label='low pass filter', lw=1, zorder=-10)

        ax.text(eod, 1.1, 'EOD=%.1fHz' % eod, rotation=-30, horizontalalignment='right',
                verticalalignment='bottom', fontsize=6)
        ax.text(eod * 2, 0.7, '2 EOD=%.1fHz' % (2 * eod), rotation=-30, horizontalalignment='right',
                verticalalignment='bottom', fontsize=6)
        ax.text(eod2, 0.8, 'stimulus=%.1fHz' % eod2, rotation=-30, horizontalalignment='right',
                verticalalignment='bottom', fontsize=6)
        ax.set_ylim((0, 2))
        ax.set_yticks([])
        ax.set_ylabel('spectrum of\nstimulus s(t)')
        ax.legend(loc='upper right')

        ax.set_xlim((0, f_max))

    def plot_membrane_potential_spectrum(self, key, ax, f_max=2000):
        dt = (self & key).fetch1['dt']
        eod = (EODFit() & key).fetch1['fundamental']
        eod2 = eod + (Runs() & key).fetch1['delta_f']

        membrane_potential = (PUnitSimulations.StimulusMembranePotential() & key).fetch1['potential']
        w = np.fft.fftfreq(len(membrane_potential), d=dt)
        idx = (w > 0) & (w < f_max)

        M = np.abs(np.fft.fft(membrane_potential))
        M /= M[idx].max()
        ax.fill_between(w[idx], 0 * w[idx], M[idx], color='dodgerblue')
        ax.set_ylim((0, 1.5))
        ax.text(eod - eod2, 0.7, 'beat=%.1fHz' % (eod - eod2), rotation=30, horizontalalignment='left',
                verticalalignment='bottom', fontsize=6)
        ax.text(eod + eod2, 0.3, '2 EOD - beat=%.1fHz' % (eod + eod2), rotation=30, horizontalalignment='left',
                verticalalignment='bottom', fontsize=6)
        ax.set_yticks([])
        ax.set_ylabel('spectrum of\nLIF input z(t)')

    def plot_spike_spectrum(self, key, ax, f_max=2000):
        df = (Runs() & key).fetch1['delta_f']

        eod = (EODFit() & key).fetch1['fundamental']
        eod2 = eod + df

        w, vs, ci = (PUnitSimulations.StimulusSecondOrderSpectrum() & key).fetch1['freq', 'spectrum', 'ci']
        stimulus_spikes = (PUnitSimulations.StimulusSpikes() & key).fetch['times']
        idx = (w > 0) & (w < f_max)

        ax.set_ylim((0, .8))
        ax.set_yticks(np.arange(0, 1, .4))

        ax.fill_between(w[idx], 0 * w[idx], vs[idx], color='dodgerblue')
        ci = second_order_critical_vector_strength(stimulus_spikes)
        ax.fill_between(w[idx], 0 * w[idx], 0 * w[idx] + ci, color='silver', alpha=.5)
        ax.text(2 * eod - eod2, 0.25, 'EOD + beat=%.1fHz' % (2 * eod - eod2), rotation=30, horizontalalignment='left',
                verticalalignment='bottom', fontsize=6)
        ax.set_ylabel('vector strength spectrum')

    def plot_isi(self, key, ax):
        eod = (EODFit() & key).fetch1['fundamental']
        period = 1 / eod
        baseline_spikes = (PUnitSimulations.BaselineSpikes() & key).fetch['times']
        isi = np.hstack((np.diff(r) for r in baseline_spikes))
        ax.hist(isi, bins=320, lw=0, color=sns.xkcd_rgb['charcoal grey'])
        ax.set_xlim((0, 20 * period))
        ax.set_xticks(np.arange(0, 25, 5) * period)
        ax.set_xticklabels(np.arange(0, 25, 5))
        ax.set_label('time [EOD cycles]')


@schema
@gitlog
class RandomTrials(dj.Lookup):
    definition = """
    n_total                 : int # total number of trials
    repeat_id               : int # repeat number
    ---

    """

    class TrialSet(dj.Part):
        definition = """
        ->RandomTrials
        new_trial_id            : int # index of the particular trial
        ->Runs.SpikeTimes
        ---
        """

    class CenterPhaseSet(dj.Part):
        definition = """
        ->RandomTrials
        new_trial_id       : int   # index of the phase sample
        ---
        ->EFishes
        """

    class UncenterPhaseSet(dj.Part):
        definition = """
        ->RandomTrials
        new_trial_id       : int   # index of the phase sample
        ---
        ->EFishes
        """

    def _prepare(self):
        lens = [len(self & dict(n_total=ntot)) == 10 for ntot in (100,)]
        n_total = 100
        if not np.all(lens):
            data = (Runs() * Runs.SpikeTimes() & dict(contrast=20, cell_id="2014-12-03-ad",
                                                      delta_f=-400)).project().fetch.as_dict()
            data = list(sorted(data, key=lambda x: x['trial_id']))
            n = len(data)

            df_center = pd.DataFrame(CenteredPUnitPhases().fetch[dj.key])
            df_uncenter = pd.DataFrame(UncenteredPUnitPhases().fetch[dj.key])

            ts = self.TrialSet()
            cps = self.CenterPhaseSet()
            ups = self.UncenterPhaseSet()

            for repeat_id in range(10):
                key = dict(n_total=n_total, repeat_id=repeat_id)
                self.insert1(key, skip_duplicates=True)
                for new_trial_id, trial_id in enumerate(np.random.randint(n, size=n_total)):
                    key['new_trial_id'] = new_trial_id
                    key.update(data[trial_id])
                    ts.insert1(key)

                key = dict(n_total=n_total, repeat_id=repeat_id)
                for new_trial_id, ix in enumerate(np.random.randint(len(df_center), size=n_total)):
                    key['new_trial_id'] = new_trial_id
                    key.update(df_center.iloc[ix].to_dict())
                    cps.insert1(key)

                    key.update(df_uncenter.iloc[ix].to_dict())
                    ups.insert1(key)

    def load_spikes(self, key, centered=True):
        trials = ((LocalEODPeaksTroughs() * Runs.SpikeTimes() \
                   * RandomTrials.TrialSet() \
                   * (RandomTrials.PhaseSet() * CenteredPUnitPhases()).project('phase', phase_cell='cell_id')) & key)

        dt = 1. / (Runs() & trials).fetch1['samplingrate']
        eod, duration = (Runs() & trials).fetch1['eod', 'duration']
        rad2period = 1 / 2 / np.pi / eod
        # get spikes, convert to s, align to EOD, add bootstrapped phase

        spikes = [s / 1000 - p[0] * dt + ph * rad2period for s, p, ph in zip(*trials.fetch['times', 'peaks', 'phase'])]
        return spikes, dt, eod, duration


@schema
@gitlog
class PyramidalSimulationParameters(dj.Lookup):
    definition = """
    pyr_simul_id    : tinyint
    ---
    tau_synapse     : double    # time constant of the synapse
    tau_neuron      : double    # time constant of the lif
    n               : int       # how many trials to simulate
    noisesd         : double    # noise standard deviation
    amplitude       : double    # multiplicative factor on the input
    offset          : double    # additive factor on the input
    threshold       : double    # LIF threshold
    reset           : double    # reset potential
    """

    contents = [
        dict(pyr_simul_id=0, tau_synapse=0.001, tau_neuron=0.01, n=1000, noisesd=30,
             amplitude=1.7, threshold=15, reset=0, offset=-20),
    ]


@schema
@gitlog
class PyramidalLIF(dj.Computed):
    definition = """
    ->RandomTrials
    ->PyramidalSimulationParameters
    centered        : bool  # whether the phases got centered per fish
    ---
    """

    class SpikeTimes(dj.Part):
        definition = """
        ->PyramidalLIF
        simul_trial_id  :   smallint # trial number
        ---
        times           : longblob  # spike times in s
        """

    def _make_tuples(self, key):
        for centered in [True, False]:
            # load spike trains for randomly selected trials
            data, dt, eod, duration = RandomTrials().load_spikes(key, centered=centered)
            key['centered'] = centered

            eod_period = 1 / eod

            # compute stimulus frequency
            delta_f = np.unique((Runs() * RandomTrials() * RandomTrials.TrialSet() & key).fetch['delta_f'])
            if len(delta_f) > 1:
                raise ValueError('delta_f should be unique')
            else:
                delta_f = delta_f.squeeze()
            stim_freq = eod + delta_f

            # get parameters for simulation
            params = (PyramidalSimulationParameters() & key).fetch1()
            params.pop('pyr_simul_id')
            print('Parameters', key)

            # plot histogram of jittered data
            fig, ax = plt.subplots()
            ax.hist(np.hstack(data) % (1 / eod), bins=100)
            fig.savefig('punitinput_{repeat_id}_{centered}.png'.format(**key))
            plt.close(fig)

            # convolve with exponential filter
            tau_s = params.pop('tau_synapse')
            bins = np.arange(0, duration + dt, dt)
            t = np.arange(0, 10 * tau_s, dt)
            h = np.exp(-np.abs(t) / tau_s)
            trials = np.vstack([np.convolve(np.histogram(sp, bins=bins)[0], h, 'full') for sp in data])[:, :-len(h) + 1]

            # simulate neuron
            t = np.arange(0, duration, dt)
            ret, V = simple_lif(t, trials.sum(axis=0),
                                **params)  # TODO make that mean to be independent of number of neurons

            isi = [np.diff(r) for r in ret]
            fig, ax = plt.subplots()
            ax.hist(np.hstack(isi), bins=100)
            ax.set_xticks(eod_period * np.arange(0, 50, 10))
            ax.set_xticklabels(np.arange(0, 50, 10))
            fig.savefig('pyr_isi_{repeat_id}_{centered}.png'.format(**key))
            plt.close(fig)

            sisi = np.hstack(isi)
            print('Firing rates (min, max, avg)', (1 / sisi).min(), (1 / sisi).max(), np.mean([len(r) for r in ret]))

            self.insert1(key)
            st = self.SpikeTimes()
            for i, trial in enumerate(ret):
                key['simul_trial_id'] = i
                key['times'] = trial
                st.insert1(key)


def simple_lif(t, I, n=10, offset=0, amplitude=1, noisesd=30, threshold=15, reset=0, tau_neuron=0.01):
    dt = t[1] - t[0]

    I = amplitude * I + offset

    Vout = np.ones(n) * reset

    ret = [list() for _ in range(n)]

    sdB = np.sqrt(dt) * noisesd
    V = np.zeros((n, len(I)))
    for i, t_step in enumerate(t):
        Vout += (-Vout + I[i]) * dt / tau_neuron + np.random.randn(n) * sdB
        idx = Vout > threshold
        for j in np.where(idx)[0]:
            ret[j].append(t_step)
        Vout[idx] = reset
        V[:, i] = Vout
    return ret, V


if __name__ == '__main__':
    EODFit().populate(reserve_jobs=True)
    LIFPUnit().populate(reserve_jobs=True)
    PUnitSimulations().populate(reserve_jobs=True)
