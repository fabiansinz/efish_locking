from locking import analyses as ana


ana.TrialAlign().populate(reserve_jobs=True)
ana.FirstOrderSpikeSpectra().populate(reserve_jobs=True)
ana.FirstOrderSignificantPeaks().populate(reserve_jobs=True)
ana.SecondOrderSpikeSpectra().populate(reserve_jobs=True)
ana.SecondOrderSignificantPeaks().populate(reserve_jobs=True)
ana.StimulusSpikeJitter().populate(reserve_jobs=True)
ana.PhaseLockingHistogram().populate(reserve_jobs=True)
ana.EODStimulusPSTSpikes().populate(reserve_jobs=True)
ana.Decoding().populate(reserve_jobs=True)
ana.BaselineSpikeJitter().populate(reserve_jobs=True)

ana.TrialAlign().progress()
ana.FirstOrderSpikeSpectra().progress()
ana.FirstOrderSignificantPeaks().progress()
ana.SecondOrderSpikeSpectra().progress()
ana.SecondOrderSignificantPeaks().progress()
ana.StimulusSpikeJitter().progress()
ana.PhaseLockingHistogram().progress()
ana.EODStimulusPSTSpikes().progress()
ana.Decoding().progress()
ana.BaselineSpikeJitter().progress()

