import modelling as mod
import seaborn as sns

sns.set_context('poster')
fig, ax = mod.PyramidalLIF().plot_vector_strength()

fig.savefig('figures/poster/pyramidal_locking_simulation.pdf')