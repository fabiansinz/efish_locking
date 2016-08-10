from locking import analyses as ana


ana.TrialAlign().populate(reserve_jobs=True)
ana.FirstOrderSpikeSpectra().populate(reserve_jobs=True)
ana.FirstOrderSignificantPeaks().populate(reserve_jobs=True)
ana.StimulusSpikeJitter().populate(reserve_jobs=True)
ana.PhaseLockingHistogram().populate(reserve_jobs=True)
ana.EODStimulusPSTSpikes().populate(reserve_jobs=True)
ana.Decoding().populate(reserve_jobs=True)
ana.BaselineSpikeJitter().populate(reserve_jobs=True)