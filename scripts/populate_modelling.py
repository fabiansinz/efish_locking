from locking import modelling as mod

print('Populating EODFit')
mod.EODFit().populate(reserve_jobs=True)
print('Populating LIFPUnit')
mod.LIFPUnit().populate(reserve_jobs=True)
print('Populating PUnitSimulations')
mod.PUnitSimulations().populate(reserve_jobs=True)
print('Populating PyramidalLIF')
mod.PyramidalLIF().populate(reserve_jobs=False)
print('Populating LIFStimulusLocking')
mod.LIFStimulusLocking().populate(reserve_jobs=True)
