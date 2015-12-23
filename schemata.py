import os
import re
import datajoint as dj
from datajoint import schema
import sys
import yaml
import seaborn as sns

BASEDIR = '/home/fabee/data/carolin/'
server = schema('efish_locking', locals())
from pyrelacs.DataClasses import load
import numpy as np
from pint import UnitRegistry
import pycircstat as circ
ureg = UnitRegistry()


def peakdet(v, delta=None):
    sys.stdout.write('.')
    maxtab = []
    maxidx = []

    mintab = []
    minidx = []
    v = np.asarray(v)
    if delta is None:
        up = int(np.min([1e5, len(v)]))
        tmp = np.abs(v[:up])
        delta = np.percentile(tmp, 99.9) - np.percentile(tmp, 50)

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True
    n = len(v)
    for i in range(len(v)):
        # if i % 1e3 == 0:
        # sys.stdout.write("\r\t\t%.2f%%" % (float(i)/float(n)*100.,))
        # sys.stdout.write("\r%i/%i" % (i, n))
        this = v[i]
        if this > mx:
            mx = this
            mxpos = i

        if this < mn:
            mn = this
            mnpos = i

        if lookformax:
            if this < mx - delta:
                maxtab.append(mx)
                maxidx.append(mxpos)
                mn = this
                mnpos = i
                lookformax = False

        else:
            if this > mn + delta:
                mintab.append(mn)
                minidx.append(mnpos)
                mx = this
                mxpos = i
                lookformax = True

    return np.asarray(maxtab), np.asarray(maxidx, dtype=int), np.asarray(mintab), np.asarray(minidx, dtype=int)


def scan_info(cell_id):
    """
    Scans the info.dat for meta information about the recording.

    :param cell_id: id of the cell
    :return: meta information about the recording as a dictionary.
    """
    info = open(BASEDIR + cell_id + '/info.dat').readlines()
    info = [re.sub(r'[^\x00-\x7F]+', ' ', e[1:]) for e in info]
    meta = yaml.load(''.join(info))
    return meta


def get_number_and_unit(value_string):
    if value_string.endswith('%'):
        return (float(value_string.strip()[:-1]), '%')
    try:
        a = ureg.parse_expression(value_string)
    except:
        return (value_string, None)

    if type(a) == list:
        return (value_string, None)

    if isinstance(a, (int, float)):
        return (a, None)
    else:
        # a.ito_base_units()
        value = a.magnitude
        unit = "{:~}".format(a)
        unit = unit[unit.index(" "):].replace(" ", "")

        if unit == 'min':
            unit = 's'
            value *= 60
        return (value, unit)


def load_traces(relacsdir, stimuli):
    meta, key, data = stimuli.selectall()

    ret = []
    for _, name, _, data_type, index in [k for k in key[0] if 'traces' in k]:
        tmp = {}
        sample_interval, time_unit = get_number_and_unit(meta[0]['analog input traces']['sample interval%i' % (index,)])
        sample_unit = meta[0]['analog input traces']['unit%i' % (index,)]
        x = np.fromfile('%s/trace-%i.raw' % (relacsdir, index), np.float32)

        tmp['unit'] = sample_unit
        tmp['trace_data'] = name
        tmp['data'] = x
        tmp['sample_interval'] = sample_interval
        tmp['sample_unit'] = sample_unit
        ret.append(tmp)
    return {e['trace_data']: e for e in ret}


@server
class PaperCells(dj.Lookup):
    definition = """
    # What cell ids make it into the paper

    cell_id                          : varchar(40)      # unique cell id
    ---
    """

    def __init__(self):
        dj.Lookup.__init__(self)

    contents = [{'cell_id': '2014-12-03-al'},
                {'cell_id': '2014-07-23-ae'},
                {'cell_id': '2014-12-11-ae-invivo-1'},
                {'cell_id': '2014-12-11-aj-invivo-1'},
                {'cell_id': '2014-12-11-al-invivo-1'},
                {'cell_id': '2014-09-02-ad'},
                {'cell_id': '2014-12-03-ab'},
                {'cell_id': '2014-07-23-ai'},
                {'cell_id': '2014-12-03-af'},
                {'cell_id': '2014-07-23-ab'},
                {'cell_id': '2014-07-23-ah'},
                {'cell_id': '2014-11-13-aa'},
                {'cell_id': '2014-11-26-ab'},
                {'cell_id': '2014-11-26-ad'},
                {'cell_id': '2014-10-29-aa'},
                {'cell_id': '2014-12-03-ae'},
                {'cell_id': '2014-09-02-af'},
                {'cell_id': '2014-12-11-ah-invivo-1'},
                {'cell_id': '2014-07-23-ad'},
                {'cell_id': '2014-12-11-ac-invivo-1'},
                {'cell_id': '2014-12-11-ab-invivo-1'},
                {'cell_id': '2014-12-11-ag-invivo-1'},
                {'cell_id': '2014-12-03-ad'},
                {'cell_id': '2014-12-11-ak-invivo-1'},
                {'cell_id': '2014-09-02-ag'},
                {'cell_id': '2014-12-03-ao'},
                {'cell_id': '2014-12-03-aj'},
                {'cell_id': '2014-07-23-aj'},
                {'cell_id': '2014-11-26-ac'},
                {'cell_id': '2014-12-03-ai'},
                {'cell_id': '2014-06-06-ak'},
                {'cell_id': '2014-11-13-ab'},
                {'cell_id': '2014-05-21-ab'},
                {'cell_id': '2014-07-23-ag'},
                {'cell_id': '2014-12-03-ah'},
                {'cell_id': '2014-07-23-aa'},
                {'cell_id': '2014-12-11-am-invivo-1'},
                {'cell_id': '2014-12-11-aa-invivo-1'}]


@server
class EFishes(dj.Imported):
    definition = """
    # Basics weakly electric fish subject info

    fish_id                                 : varchar(40)     # unique fish id
    ->PaperCells
    ---

    eod_frequency                           : float # EOD frequency in Hz
    species = "Apteronotus leptorhynchus"   : enum('Apteronotus leptorhynchus', 'Eigenmannia virescens') # species
    gender  = "unknown"                     : enum('unknown', 'make', 'female') # gender
    weight                                  : float # weight in g
    size                                    : float  # size in cm
    """

    def _make_tuples(self, key):
        a = scan_info(key['cell_id'])
        a = a['Subject'] if 'Subject' in a else a['Recording']['Subject']
        key.update({'fish_id': a['Identifier'],
                    'eod_frequency': float(a['EOD Frequency'][:-2]),
                    'gender': a['Gender'].lower(),
                    'weight': float(a['Weight'][:-1]),
                    'size': float(a['Size'][:-2])})

        self.insert1(key)


@server
class Cells(dj.Imported):
    definition = """
    # Recorded cell with additional info

    ->PaperCells                                # cell id to be imported
    ---
    ->EFishes                                   # fish this cell was recorded from
    recording_date                   : date     #  recording date
    cell_type                        : enum('p-unit', 'i-cell', 'e-cell','ampullary','ovoid','unkown') # cell type
    recording_location               : enum('nerve', 'ell') # where
    depth                            : float    # recording depth in mu
    baseline                         : float    # baseline firing rate in Hz
    """

    def _make_tuples(self, key):
        a = scan_info(key['cell_id'])
        subj = a['Subject'] if 'Subject' in a else a['Recording']['Subject']
        cl = a['Cell'] if 'Cell' in a else a['Recording']['Cell']
        dat = {'cell_id': key['cell_id'],
               'fish_id': subj['Identifier'],
               'recording_date': a['Recording']['Date'],
               'cell_type': cl['CellType'].lower(),
               'recording_location': 'nerve' if cl['Structure'].lower() == 'nerve' else 'ell',
               'depth': float(cl['Depth'][:-2]),
               'baseline': float(cl['Cell properties']['Firing Rate1'][:-2])
               }
        self.insert1(dat)


@server
class FICurves(dj.Imported):
    definition = """
    # FI  Curves from recorded cells

    block_no            : int           # id of the fi curve block
    ->Cells                             # cell ids

    ---

    inx                 : longblob      # index
    n                   : longblob      # no of repeats
    ir                  : longblob      # Ir in mV
    im                  : longblob      # Im in mV
    f_0                 : longblob      # f0 in Hz
    f_s                 : longblob      # fs in Hz
    f_r                 : longblob      # fr in Hz
    ip                  : longblob      # Ip in mV
    ipm                 : longblob      # Ipm in mV
    f_p                 : longblob      # fp in Hz
    """

    def _make_tuples(self, key):
        filename = BASEDIR + key['cell_id'] + '/ficurves1.dat'
        if os.path.isfile(filename):
            fi = load(filename)

            for i, (fi_meta, fi_key, fi_data) in enumerate(zip(*fi.selectall())):

                fi_data = np.asarray(fi_data).T
                row = {'block_no': i, 'cell_id': key['cell_id']}
                for (name, _), dat in zip(fi_key, fi_data):
                    row[name.lower()] = dat

                self.insert1(row)

    def plot(self, ax, restrictions):
        rel = self & restrictions
        try:
            contrast, f0, fs = (self & restrictions).fetch1['ir', 'f_0', 'f_s']
        except dj.DataJointError:
            return
        ax.plot(contrast, f0, '--k', label='onset response', dashes=(2, 2))
        ax.plot(contrast, fs, '-k', label='stationary response')
        ax.set_xlabel('amplitude [mV/cm]')
        ax.set_ylabel('firing rate [Hz]')
        _, ymax = ax.get_ylim()
        ax.set_ylim((0, 1.5 * ymax))
        mi, ma = np.amin(contrast), np.amax(contrast)
        ax.set_xticks(np.round([mi, (ma + mi) * .5, ma], decimals=1))


@server
class ISIHistograms(dj.Imported):
    definition = """
    # ISI Histograms

    block_no            : int           # id of the isi curve block
    ->Cells                             # cell ids

    ---

    t                   : longblob      # time
    n                   : longblob      # no of repeats
    eod                 : longblob      # time in eod cycles
    p                   : longblob      # histogram
    """

    def _make_tuples(self, key):
        filename = BASEDIR + key['cell_id'] + '/baseisih1.dat'
        if os.path.isfile(filename):
            fi = load(filename)

            for i, (fi_meta, fi_key, fi_data) in enumerate(zip(*fi.selectall())):

                fi_data = np.asarray(fi_data).T
                row = {'block_no': i, 'cell_id': key['cell_id']}
                for (name, _), dat in zip(fi_key, fi_data):
                    row[name.lower()] = dat

                self.insert1(row)

    def plot(self, ax, restrictions):
        try:
            eod_cycles, p = (ISIHistograms() & restrictions).fetch1['eod', 'p']
        except dj.DataJointError:
            return
        dt = eod_cycles[1] - eod_cycles[0]
        ax.bar(eod_cycles, p, width=dt, color=sns.xkcd_rgb['charcoal grey'], lw=0)
        ax.set_xlabel('EOD cycles')

@server
class Baseline(dj.Imported):
    definition = """
    # table holding baseline recordings
    ->Cells
    repeat                     : int # index of the run

    ---
    eod                     : float # eod rate at trial in Hz
    duration                : float # duration in s
    samplingrate            : float # sampling rate in Hz

    """

    class SpikeTimes(dj.Part):
        definition = """
        # table holding spike time of trials

        -> Baseline
        ---

        times                      : longblob # spikes times in ms
        """

    class LocalEODPeaksTroughs(dj.Part, dj.Manual):
        definition = """
        # table holding local EOD traces

        -> Baseline
        ---

        peaks                      : longblob
        troughs                      : longblob
        """

    def mean_var(self):
        """
        Computes the mean and variance of the baseline psth

        :return: mean and variance
        """
        spikes = (Baseline.SpikeTimes() & self).fetch1['times']
        eod = self.fetch1['eod']
        period = 1/eod
        factor = 2 * np.pi / period
        t = (spikes % period)
        mu = circ.mean(t * factor)/factor
        sigma2 = circ.var(t * factor) / factor**2
        return mu, sigma2

    def _make_tuples(self, key):
        repro = 'BaselineActivity'
        basedir = BASEDIR + key['cell_id']
        spikefile = basedir + '/basespikes1.dat'
        if os.path.isfile(spikefile):
            stimuli = load(basedir + '/stimuli.dat')

            traces = load_traces(basedir, stimuli)
            spikes = load(spikefile)
            spi_meta, spi_key, spi_data = spikes.selectall()

            localeod = Baseline.LocalEODPeaksTroughs()
            spike_table = Baseline.SpikeTimes()


            for run_idx, (spi_d, spi_m) in enumerate(zip(spi_data, spi_meta)):
                print("\t%s repeat %i" % (repro, run_idx))

                # match index from stimspikes with run from stimuli.dat
                stim_m, stim_k, stim_d = stimuli.subkey_select(RePro=repro, Run=spi_m['index'])


                if len(stim_m) > 1:
                    raise KeyError('%s and index are not unique to identify stimuli.dat block.' % (repro,))
                else:
                    stim_k = stim_k[0]
                    stim_m = stim_m[0]
                    signal_column = \
                        [i for i, k in enumerate(stim_k) if k[:4] == ('stimulus', 'GlobalEField', 'signal', '-')][0]

                    valid = []

                    if stim_d == [[[0]]]:
                        print("\t\tEmpty stimuli data! Continuing ...")
                        continue


                    for d in stim_d[0]:
                        if not d[signal_column].startswith('FileStimulus-value'):
                            valid.append(d)
                        else:
                            print("\t\tExcluding a reset trial from stimuli.dat")
                    stim_d = valid

                if len(stim_d) > 1:
                    print(
                        """\t\t%s index %i has more one trials. Not including data.""" % (
                            spikefile, spi_m['index'], len(spi_d), len(stim_d)))
                    continue

                start_index, index = [(i, k[-1]) for i, k in enumerate(stim_k) if 'traces' in k and 'V-1' in k][0]
                sample_interval, time_unit = get_number_and_unit(
                    stim_m['analog input traces']['sample interval%i' % (index,)])

                # make sure that everything was sampled with the same interval
                sis = []
                for jj in range(1, 5):
                    si, tu = get_number_and_unit(stim_m['analog input traces']['sample interval%i' % (jj,)])
                    assert tu == 'ms', 'Time unit is not ms anymore!'
                    sis.append(si)
                assert len(np.unique(sis)) == 1, 'Different sampling intervals!'

                duration = ureg.parse_expression(spi_m['duration']).to(time_unit).magnitude



                start_idx, stop_idx = [], []
                # start_times, stop_times = [], []

                start_indices = [d[start_index] for d in stim_d]
                for begin_index, trial in zip(start_indices, spi_d):
                    start_idx.append(begin_index)
                    stop_idx.append(begin_index + duration / sample_interval)

                to_insert = dict(key)
                to_insert['repeat'] = spi_m['index']
                to_insert['eod'] = float(spi_m['EOD rate'][:-2])
                to_insert['duration'] = duration / 1000 if time_unit == 'ms' else duration
                to_insert['samplingrate'] = 1 / sample_interval * 1000 if time_unit == 'ms' else 1 / sample_interval

                self.insert1(to_insert)

                for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
                    if start > 0:
                        tmp = dict(key, repeat=spi_m['index'])
                        leod = traces['LocalEOD-1']['data'][start:stop]
                        _, tmp['peaks'], _, tmp['troughs'] = peakdet(leod)
                        localeod.insert1(tmp, replace=True)
                    else:
                        print("Negative indices in stimuli.dat. Skipping local peak extraction!")


                    spike_table.insert1(dict(key, times=spi_d, repeat=spi_m['index']), replace=True)

@server
class Runs(dj.Imported):
    definition = """
    # table holding trials

    run_id                     : int # index of the run
    repro="SAM"                : enum('SAM', 'Filestimulus')
    ->Cells                    # which cell the trial belongs to

    ---

    delta_f                 : float # delta f of the trial in Hz
    contrast                : float  # contrast of the trial
    eod                     : float # eod rate at trial in Hz
    duration                : float # duration in s
    am                      : int   # whether AM was used
    samplingrate            : float # sampling rate in Hz
    n_harmonics             : int # number of harmonics in the stimulus
    """

    class SpikeTimes(dj.Part):
        definition = """
        # table holding spike time of trials

        -> Runs
        trial_id                   : int # index of the trial within run

        ---

        times                      : longblob # spikes times in ms
        """

    class GlobalEField(dj.Part, dj.Manual):
        definition = """
        # table holding global efield trace

        -> Runs
        trial_id                   : int # index of the trial within run
        ---

        global_efield                      : longblob # spikes times
        """

    class LocalEOD(dj.Part, dj.Manual):
        definition = """
        # table holding local EOD traces

        -> Runs
        trial_id                   : int # index of the trial within run
        ---

        local_efield                      : longblob # spikes times
        """

    class GlobalEOD(dj.Part, dj.Manual):
        definition = """
        # table holding global EOD traces

        -> Runs
        trial_id                   : int # index of the trial within run
        ---

        global_voltage                      : longblob # spikes times
        """

    class VoltageTraces(dj.Part, dj.Manual):
        definition = """
        # table holding voltage traces

        -> Runs
        trial_id                   : int # index of the trial within run
        ---

        membrane_potential                      : longblob # spikes times
        """

    def _make_tuples(self, key):
        repro = 'SAM'
        basedir = BASEDIR + key['cell_id']
        spikefile = basedir + '/samallspikes1.dat'
        if os.path.isfile(spikefile):
            stimuli = load(basedir + '/stimuli.dat')
            traces = load_traces(basedir, stimuli)
            spikes = load(spikefile)
            spi_meta, spi_key, spi_data = spikes.selectall()

            globalefield = Runs.GlobalEField()
            localeod = Runs.LocalEOD()
            globaleod = Runs.GlobalEOD()
            spike_table = Runs.SpikeTimes()
            v1trace = Runs.VoltageTraces()

            for run_idx, (spi_d, spi_m) in enumerate(zip(spi_data, spi_meta)):
                print("\t%s run %i" % (repro, run_idx))

                # match index from stimspikes with run from stimuli.dat
                stim_m, stim_k, stim_d = stimuli.subkey_select(RePro=repro, Run=spi_m['index'])

                if len(stim_m) > 1:
                    raise KeyError('%s and index are not unique to identify stimuli.dat block.' % (repro,))
                else:
                    stim_k = stim_k[0]
                    stim_m = stim_m[0]
                    signal_column = \
                        [i for i, k in enumerate(stim_k) if k[:4] == ('stimulus', 'GlobalEField', 'signal', '-')][0]

                    valid = []

                    if stim_d == [[[0]]]:
                        print("\t\tEmpty stimuli data! Continuing ...")
                        continue

                    for d in stim_d[0]:
                        if not d[signal_column].startswith('FileStimulus-value'):
                            valid.append(d)
                        else:
                            print("\t\tExcluding a reset trial from stimuli.dat")
                    stim_d = valid

                if len(stim_d) != len(spi_d):
                    print(
                        """\t\t%s index %i has %i trials, but stimuli.dat has %i. Trial was probably aborted. Not including data.""" % (
                            spikefile, spi_m['index'], len(spi_d), len(stim_d)))
                    continue

                start_index, index = [(i, k[-1]) for i, k in enumerate(stim_k) if 'traces' in k and 'V-1' in k][0]
                sample_interval, time_unit = get_number_and_unit(
                    stim_m['analog input traces']['sample interval%i' % (index,)])

                # make sure that everything was sampled with the same interval
                sis = []
                for jj in range(1, 5):
                    si, tu = get_number_and_unit(stim_m['analog input traces']['sample interval%i' % (jj,)])
                    assert tu == 'ms', 'Time unit is not ms anymore!'
                    sis.append(si)
                assert len(np.unique(sis)) == 1, 'Different sampling intervals!'

                duration = ureg.parse_expression(spi_m['Settings']['Stimulus']['duration']).to(time_unit).magnitude

                if 'ampl' in spi_m['Settings']['Stimulus']:
                    nharmonics = len(list(map(float, spi_m['Settings']['Stimulus']['ampl'].strip().split(','))))
                else:
                    nharmonics = 0

                start_idx, stop_idx = [], []
                # start_times, stop_times = [], []

                start_indices = [d[start_index] for d in stim_d]
                for begin_index, trial in zip(start_indices, spi_d):
                    # start_times.append(begin_index*sample_interval)
                    # stop_times.append(begin_index*sample_interval + duration)
                    start_idx.append(begin_index)
                    stop_idx.append(begin_index + duration / sample_interval)

                to_insert = dict(key)
                to_insert['run_id'] = spi_m['index']
                to_insert['delta_f'] = float(spi_m['Settings']['Stimulus']['deltaf'][:-2])
                to_insert['contrast'] = float(spi_m['Settings']['Stimulus']['contrast'][:-1])
                to_insert['eod'] = float(spi_m['EOD rate'][:-2])
                to_insert['duration'] = duration / 1000 if time_unit == 'ms' else duration
                to_insert['am'] = spi_m['Settings']['Stimulus']['am'] * 1
                to_insert['samplingrate'] = 1 / sample_interval * 1000 if time_unit == 'ms' else 1 / sample_interval
                to_insert['n_harmonics'] = nharmonics
                to_insert['repro'] = 'SAM'

                self.insert1(to_insert)
                for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
                    tmp = dict(run_id=run_idx, trial_id=trial_idx, repro='SAM', **key)
                    tmp['membrane_potential'] = traces['V-1']['data'][start:stop]
                    v1trace.insert1(tmp, replace=True)
                    del tmp['membrane_potential']

                    tmp['global_efield'] = traces['GlobalEFie']['data'][start:stop]
                    globalefield.insert1(tmp, replace=True)
                    del tmp['global_efield']

                    tmp['local_efield'] = traces['LocalEOD-1']['data'][start:stop]
                    localeod.insert1(tmp, replace=True)
                    del tmp['local_efield']

                    tmp['global_voltage'] = traces['EOD']['data'][start:stop]
                    globaleod.insert1(tmp, replace=True)
                    del tmp['global_voltage']

                    tmp['times'] = spi_d[trial_idx]
                    spike_table.insert1(tmp, replace=True)


@server
class GlobalEFieldPeaksTroughs(dj.Computed):
    definition = """
    # table for peaks and troughs in the global efield

    -> Runs.GlobalEField

    ---

    peaks               : longblob # peak indices
    troughs             : longblob # trough indices
    """

    def _make_tuples(self, key):
        dat = (Runs.GlobalEField() & key).fetch1()

        _, key['peaks'], _, key['troughs'] = peakdet(dat['global_efield'])
        self.insert1(key)


@server
class LocalEODPeaksTroughs(dj.Computed):
    definition = """
    # table for peaks and troughs in local EOD

    -> Runs.LocalEOD

    ---

    peaks               : longblob # peak indices
    troughs             : longblob # trough indices
    """

    def _make_tuples(self, key):
        dat = (Runs.LocalEOD() & key).fetch1()

        _, key['peaks'], _, key['troughs'] = peakdet(dat['local_efield'])
        self.insert1(key)


if __name__ == "__main__":
    pc = PaperCells()

    ef = EFishes()
    ef.populate(reserve_jobs=True)

    cl = Cells()
    cl.populate(reserve_jobs=True)

    fi = FICurves()
    fi.populate(reserve_jobs=True)

    isi = ISIHistograms()
    isi.populate(reserve_jobs=True)

    sams = Runs()
    sams.populate(restriction=cl, reserve_jobs=True)

    pts = GlobalEFieldPeaksTroughs()
    pts.populate(reserve_jobs=True)

    lpts = LocalEODPeaksTroughs()
    lpts.populate(reserve_jobs=True)

    Baseline().populate()