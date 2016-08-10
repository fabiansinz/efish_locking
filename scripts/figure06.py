import seaborn as sns
from matplotlib.collections import PolyCollection
from locking import modelling as mod
import matplotlib.pyplot as plt

sns.set_context('paper')
with sns.axes_style("ticks"):
    fig, ax = plt.subplots(figsize=(7,5))
mod.LIFFirstOrderSpikeSpectra().plot_avg_spectrum(ax=ax, centered=True)

# for art in ax.get_children():
#     if isinstance(art, PolyCollection):
#         art.set_linewidth(0)
# fig.tight_layout()
# fig.savefig('figures/figure06.pdf')