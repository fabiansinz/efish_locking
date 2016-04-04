import seaborn as sns

from locking import modelling as mod

sns.set_context('poster')
fig, ax = mod.PyramidalLIF().plot_vector_strength()

fig.savefig('figures/poster/pyramidal_locking_simulation.pdf')