import os
from pprint import pprint
import re
import datajoint as dj
from datajoint import schema
import glob
import sys
import yaml

BASEDIR = '/home/fabee/data/carolin/'
server = schema('efish', locals())
from pyrelacs.DataClasses import load
import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()

def peakdet(v, delta=None):
    sys.stdout.write('.')
    maxtab = []
    maxidx = []

    mintab = []
    minidx = []
    v = np.asarray(v)
    if delta is None:
        up = int(np.min([1e5,len(v)]))
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
class EFishes(dj.Manual):
    definition = """
    # Basics weakly electric fish subject info

    fish_id                                 : varchar(40)     # unique fish id
    ---

    eod_frequency                           : float # EOD frequency in Hz
    species = "Apteronotus leptorhynchus"   : enum('Apteronotus leptorhynchus', 'Eigenmannia virescens') # species
    gender  = "unknown"                      : enum('unknown', 'make', 'female') # gender
    weight : float # weight in g
    size: float  # size in cm
    """

    def make_tuples(self):
        for (cell,) in PaperCells().fetch():
            a = scan_info(cell)
            a = a['Subject'] if 'Subject' in a else a['Recording']['Subject']
            dat = {'fish_id': a['Identifier'],
                   'eod_frequency': float(a['EOD Frequency'][:-2]),
                   'gender': a['Gender'].lower(),
                   'weight': float(a['Weight'][:-1]),
                   'size': float(a['Size'][:-2])}
            try:
                self.insert(dat)
            except Exception as e:
                print(e)


@server
class PaperCells(dj.Lookup):
    definition = """
    # What cell ids make it into the paper

    cell_id                          : varchar(40)      # unique cell id
    ---
    """

    def __init__(self):
        dj.Lookup.__init__(self)

    def make_tuples(self):
        cells = [{'cell_id': '2014-12-03-al'},
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
        for c in cells:
            try:
                self.insert(c)
            except Exception as e:
                print(e)


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

    @property
    def populate_relation(self):
        return PaperCells()

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
        self.insert(dat)


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

    @property
    def populate_relation(self):
        return Cells().project()

    def _make_tuples(self, key):
        filename = BASEDIR + key['cell_id'] + '/ficurves1.dat'
        if os.path.isfile(filename):
            fi = load(filename)

            for i, (fi_meta, fi_key, fi_data) in enumerate(zip(*fi.selectall())):

                fi_data = np.asarray(fi_data).T
                row = {'block_no': i, 'cell_id': key['cell_id']}
                for (name, _), dat in zip(fi_key, fi_data):
                    row[name.lower()] = dat

                self.insert(row)


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

    @property
    def populate_relation(self):
        return Cells().project()

    def _make_tuples(self, key):
        filename = BASEDIR + key['cell_id'] + '/baseisih1.dat'
        if os.path.isfile(filename):
            fi = load(filename)

            for i, (fi_meta, fi_key, fi_data) in enumerate(zip(*fi.selectall())):

                fi_data = np.asarray(fi_data).T
                row = {'block_no': i, 'cell_id': key['cell_id']}
                for (name, _), dat in zip(fi_key, fi_data):
                    row[name.lower()] = dat

                self.insert(row)


@server
class Runs(dj.Imported):
    definition = """
    # table holding trials

    run_id                     : int # index of the run
    repro="SAM"                : enum('SAM', 'Filestimulus')
    ->Cells                          # which cell the trial belongs to

    ---

    delta_f                 : float # delta f of the trial in Hz
    contrast                : float  # contrast of the trial
    eod                     : float # eod rate at trial in Hz
    duration                : float # duration in s
    am                      : int   # whether AM was used
    samplingrate            : float # sampling rate in Hz
    n_harmonics             : int # number of harmonics in the stimulus
    """

    @property
    def populate_relation(self):
        return Cells().project()

    def _make_tuples(self, key):
        repro = 'SAM'
        basedir = BASEDIR + key['cell_id']
        spikefile = basedir + '/samallspikes1.dat'
        if os.path.isfile(spikefile):
            stimuli = load(basedir + '/stimuli.dat')
            traces = load_traces(basedir, stimuli)
            spikes = load(spikefile)
            spi_meta, spi_key, spi_data = spikes.selectall()

            globalefield = GlobalEField()
            localeod = LocalEOD()
            globaleod = GlobalEOD()
            spike_table = SpikeTimes()
            v1trace = VoltageTraces()

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

                self.insert(to_insert)
                for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
                    tmp = dict(run_id=run_idx, trial_id=trial_idx, repro='SAM', **key)
                    tmp['trace'] = traces['V-1']['data'][start:stop]
                    v1trace.insert(tmp, replace=True)

                    tmp['trace'] = traces['GlobalEFie']['data'][start:stop]
                    globalefield.insert(tmp, replace=True)

                    tmp['trace'] = traces['LocalEOD-1']['data'][start:stop]
                    localeod.insert(tmp, replace=True)

                    tmp['trace'] = traces['EOD']['data'][start:stop]
                    globaleod.insert(tmp, replace=True)

                    tmp.pop('trace')
                    tmp['times'] = spi_d[trial_idx]
                    spike_table.insert(tmp, replace=True)

@server
class SpikeTimes(dj.Subordinate, dj.Manual):
    definition = """
    # table holding spike time of trials

    -> Runs
    trial_id                   : int # index of the trial within run

    ---

    times                      : longblob # spikes times in ms
    """


@server
class GlobalEField(dj.Subordinate, dj.Manual):
    definition = """
    # table holding global efield trace

    -> Runs
    trial_id                   : int # index of the trial within run

    ---

    trace                      : longblob # spikes times
    """


@server
class LocalEOD(dj.Subordinate, dj.Manual):
    definition = """
    # table holding local EOD traces

    -> Runs
    trial_id                   : int # index of the trial within run

    ---

    trace                      : longblob # spikes times
    """


@server
class GlobalEOD(dj.Subordinate, dj.Manual):
    definition = """
    # table holding global EOD traces

    -> Runs
    trial_id                   : int # index of the trial within run

    ---

    trace                      : longblob # spikes times
    """


@server
class VoltageTraces(dj.Subordinate, dj.Manual):
    definition = """
    # table holding voltage traces

    -> Runs
    trial_id                   : int # index of the trial within run
    ---

    trace                      : longblob # spikes times
    """

@server
class GlobalEFieldPeaksTroughs(dj.Computed):
    definition = """
    # table for peaks and troughs in the global efield

    -> GlobalEField

    ---

    peaks               : longblob # peak indices
    troughs             : longblob # trough indices
    """

    @property
    def populate_relation(self):
        return GlobalEField()

    def _make_tuples(self, key):

        dat = (GlobalEField() & key).fetch(as_dict=True)
        assert len(dat) == 1, 'key returned more than one element'

        _, key['peaks'], _, key['troughs'] = peakdet(dat[0]['trace'])
        self.insert(key)

@server
class LocalEODPeaksTroughs(dj.Computed):
    definition = """
    # table for peaks and troughs in local EOD

    -> LocalEOD

    ---

    peaks               : longblob # peak indices
    troughs             : longblob # trough indices
    """

    @property
    def populate_relation(self):
        return LocalEOD()

    def _make_tuples(self, key):

        dat = (LocalEOD() & key).fetch(as_dict=True)
        assert len(dat) == 1, 'key returned more than one element'

        _, key['peaks'], _, key['troughs'] = peakdet(dat[0]['trace'])
        self.insert(key)


if __name__ == "__main__":
    # pc = PaperCells()
    # pc.make_tuples()
    #
    # ef = EFishes()
    # ef.make_tuples()
    #
    # cl = Cells()
    # cl.populate()
    #
    # fi = FICurves()
    # fi.populate()
    # print(fi)
    #
    # isi = ISIHistograms()
    # isi.populate()
    # print(isi)
    #
    # sams = Runs()
    # sams.populate(restriction=cl)

    pts = GlobalEFieldPeaksTroughs()
    pts.populate()

    lpts = LocalEODPeaksTroughs()
    lpts.populate()
