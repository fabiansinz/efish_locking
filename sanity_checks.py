from itertools import repeat
import datajoint as dj
from datajoint import schema
from schemata import Runs, LocalEODPeaksTroughs, GlobalEFieldPeaksTroughs
server = schema('efish_locking', locals())
import numpy as np


@server
class PeakTroughCheck(dj.Computed):
    definition = """
    ->Runs

    ---

    eod_frequency_peak           : double
    stimulus_frequency_peak      : double
    eod_frequency_trough           : double
    stimulus_frequency_trough      : double
    """

    class SingleEODFrequencies(dj.Part):
        definition = """
        ->PeakTroughCheck
        ->LocalEODPeaksTroughs
        ---
        freq_peak                    : double
        freq_trough                    : double
        """

    class SingleEFieldFrequencies(dj.Part):
        definition = """
        ->PeakTroughCheck
        ->GlobalEFieldPeaksTroughs
        ---
        freq_peak                    : double
        freq_trough                    : double
        """

    @property
    def populated_from(self):
        return Runs() & 'am=0'

    def _make_tuples(self, key):
        dat_eod = (Runs() * LocalEODPeaksTroughs() & key).fetch()

        eod_peaks = dat_eod['samplingrate']/np.array([np.diff(e).mean() for e in dat_eod['peaks']])
        eod_troughs = dat_eod['samplingrate']/np.array([np.diff(e).mean() for e in dat_eod['troughs']])

        dat_efield = (Runs() * GlobalEFieldPeaksTroughs() & key).fetch()
        efield_peaks = dat_efield['samplingrate']/np.array([np.diff(e).mean() for e in dat_efield['peaks']])
        efield_troughs = dat_efield['samplingrate']/np.array([np.diff(e).mean() for e in dat_efield['troughs']])

        key_sub = dict(key)
        key['eod_frequency_peak'] = eod_peaks.mean()
        key['eod_frequency_trough'] = eod_troughs.mean()
        key['stimulus_frequency_peak'] = efield_peaks.mean()
        key['stimulus_frequency_trough'] = efield_troughs.mean()

        self.insert1(key)

        n = len(dat_eod)
        PeakTroughCheck.SingleEODFrequencies().insert(zip(
                    dat_eod['run_id'],
                    dat_eod['repro'],
                    dat_eod['cell_id'],
                    dat_eod['trial_id'],
                    eod_peaks,
                    eod_troughs
        ))

        n = len(dat_efield)
        PeakTroughCheck.SingleEFieldFrequencies().insert(zip(
                    dat_efield['run_id'],
                    dat_efield['repro'],
                    dat_efield['cell_id'],
                    dat_efield['trial_id'],
                    efield_peaks,
                    efield_troughs
        ))


if __name__=="__main__":
    PeakTroughCheck().populate()