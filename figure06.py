import seaborn as sns
from matplotlib.collections import PolyCollection
from locking import modelling as mod
import matplotlib.pyplot as plt

sns.set_context('poster')
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(7,5))
mod.PyramidalLIF().plot_vector_strength(color='grey', ax=ax)
for art in ax.get_children():
    if isinstance(art, PolyCollection):
        art.set_linewidth(0)
fig.tight_layout()
fig.savefig('figures/figure06.pdf')