from pandas import DataFrame

import datajoint as dj
from datajoint import schema
from locking.analyses import FirstOrderSignificantPeaks, SecondOrderSignificantPeaks
from locking.data import Runs, LocalEODPeaksTroughs, GlobalEFieldPeaksTroughs

server = schema('efish_tests', locals())
import numpy as np
import unittest

TOL = 3

@server
class PeakTroughCheck(dj.Computed, unittest.TestCase):
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

        eod_peaks = dat_eod['samplingrate'] / np.array([np.diff(e).mean() for e in dat_eod['peaks']])
        eod_troughs = dat_eod['samplingrate'] / np.array([np.diff(e).mean() for e in dat_eod['troughs']])

        dat_efield = (Runs() * GlobalEFieldPeaksTroughs() & key).fetch()
        efield_peaks = dat_efield['samplingrate'] / np.array([np.diff(e).mean() for e in dat_efield['peaks']])
        efield_troughs = dat_efield['samplingrate'] / np.array([np.diff(e).mean() for e in dat_efield['troughs']])

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

    @property
    def inconsistent_peakdet_relacs_runs(self):
        return Runs() * self & """(ABS(eod - eod_frequency_peak) > %(tol)f) or (ABS(eod - eod_frequency_trough) > %(tol)f)
                            or (ABS(eod + delta_f - stimulus_frequency_peak) > %(tol)f)
                            or (ABS(eod + delta_f - stimulus_frequency_trough) > %(tol)f)""" % dict(tol=TOL)

    @property
    def inconsistent_peakdet_spikelocking_runs_stimulus(self):
        return FirstOrderSignificantPeaks() * self & 'baseline_coeff=0' & '(stimulus_coeff=1 and eod_coeff=0)' \
            & 'refined=1' \
            & '(ABS(frequency - stimulus_frequency_peak) > %(tol)f) or (ABS(frequency - stimulus_frequency_trough) > %(tol)f)' % dict(tol=TOL)

    @property
    def inconsistent_2nd_order_peakdet_spikelocking_runs_stimulus(self):
        return SecondOrderSignificantPeaks() * self & 'baseline_coeff=0' & '(stimulus_coeff=1 and eod_coeff=0)' \
            & 'refined=1' \
            & '(ABS(frequency - stimulus_frequency_peak) > %(tol)f) or (ABS(frequency - stimulus_frequency_trough) > %(tol)f)' % dict(tol=TOL)

    @property
    def inconsistent_peakdet_spikelocking_runs_eod(self):
        return FirstOrderSignificantPeaks() * self & 'baseline_coeff=0' & '(stimulus_coeff=0 and eod_coeff=1)' \
            & 'refined=1' \
            & '(ABS(frequency - eod_frequency_peak) > %(tol)f) or (ABS(frequency - eod_frequency_trough) > %(tol)f)' % dict(tol=TOL)

    @property
    def inconsistent_2nd_order_peakdet_spikelocking_runs_eod(self):
        return SecondOrderSignificantPeaks() * self & 'baseline_coeff=0' & '(stimulus_coeff=0 and eod_coeff=1)' \
            & 'refined=1' \
            & '(ABS(frequency - eod_frequency_peak) > %(tol)f) or (ABS(frequency - eod_frequency_trough) > %(tol)f)' % dict(tol=TOL)

    # ------------ testing --------------
    def setUp(self):
        self.populate()

    def test_relacs_peakdet_consistency(self):
        """Test whether the frequencies from relacs and peakdet are the same"""
        rel = self.inconsistent_peakdet_relacs_runs
        n = len(rel)
        df = DataFrame(rel.fetch())
        m = np.max([np.abs(df.eod - df.eod_frequency_peak), np.abs(df.eod - df.eod_frequency_trough),
                    np.abs(df.eod + df.delta_f - df.stimulus_frequency_peak),
                    np.abs(df.eod + df.delta_f - df.stimulus_frequency_trough), ])
        self.assertTrue(n == 0, '%i tuples deviate in frequency estimates by more than %iHz (maxdev %.4fHz)' % (n, TOL, m))

    def test_1st_order_eod(self):
        rel = self.inconsistent_peakdet_spikelocking_runs_eod
        n = len(rel)
        self.assertTrue(n == 0, '%i tuples deviate in eod estimates by more than %iHz' % (n, TOL))

    def test_1st_order_stimulus(self):
        rel = self.inconsistent_peakdet_spikelocking_runs_stimulus
        n = len(rel)
        self.assertTrue(n == 0, '%i tuples deviate in eod estimates by more than %iHz' % (n, TOL))

    def test_2nd_order_eod(self):
        rel = self.inconsistent_2nd_order_peakdet_spikelocking_runs_stimulus
        n = len(rel)
        self.assertTrue(n == 0, '%i tuples deviate in eod estimates by more than %iHz' % (n, TOL))

    def test_2nd_order_stimulus(self):
        rel = self.inconsistent_2nd_order_peakdet_spikelocking_runs_eod
        n = len(rel)
        self.assertTrue(n == 0, '%i tuples deviate in eod estimates by more than %iHz' % (n, TOL))
