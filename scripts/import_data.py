from locking import data, sanity
import numpy as np

data.EFishes().populate(reserve_jobs=True)
data.Cells().populate(reserve_jobs=True)
data.FICurves().populate(reserve_jobs=True)
data.ISIHistograms().populate(reserve_jobs=True)
data.Baseline().populate(reserve_jobs=True)
data.Runs().populate(reserve_jobs=True)
data.GlobalEFieldPeaksTroughs().populate(reserve_jobs=True)
data.GlobalEODPeaksTroughs().populate(reserve_jobs=True)
data.LocalEODPeaksTroughs().populate(reserve_jobs=True)
data.PUnitPhases().populate(reserve_jobs=True)

sanity.SpikeCheck().populate(reserve_jobs=True)

print('These Runs have no spikes at all and should be deleted')
print(data.Runs() * sanity.SpikeCheck() & 'all_zeros > 0')
(data.Runs() & sanity.SpikeCheck() & 'all_zeros > 0').delete()

for k in (sanity.SpikeCheck.SpikeCount() * data.Runs.SpikeTimes() & 'is_empty=1').fetch.keys():
    (data.Runs().SpikeTimes() & k)._update('times', np.array([]))

# relabel misslabeled cell
(data.Cells() & dict(cell_id='2014-05-21-ab'))._update('cell_type', 'p-unit')
