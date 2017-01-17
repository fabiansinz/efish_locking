from locking import analyses as ana
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from locking import mkdir
from scripts.plot_settings import params as plot_params, FormatedFigure
from scipy import stats


class Figure07(FormatedFigure):
    def prepare(self):
        sns.set_context('paper')
        sns.set_style('ticks')
        sns.set_palette('PuBuGn_d', n_colors=len(pd.unique(df.cell_id)))
        with plt.rc_context(plot_params):
            self.ax = {}
            self.fig = plt.figure(figsize=(6, 2.5), dpi=400)
            self.gs = plt.GridSpec(1, 20)
            self.ax['difference'] = self.fig.add_subplot(self.gs[0, 10:])
            self.ax['scatter'] = self.fig.add_subplot(self.gs[0, :10])

    @staticmethod
    def format_difference(ax):
        # ax.legend(bbox_to_anchor=(1.6, 1.05), bbox_transform=ax.transAxes, prop={'size': 6})
        ax.set_xlabel(r'$\Delta f/$EODf')
        ax.set_ylabel(r'$\nu$(stimulus) - $\nu$(AM)')
        ax.set_xlim((-.6, .6))
        ax.set_xticks(np.arange(-.5, 1, .5))
        ax.tick_params('both', length=3, width=1, which='both')

        ax.text(-0.3, 1, 'B', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_color(ax):
        ax.tick_params('y', length=0, width=0, which='both', pad=-.15)

    @staticmethod
    def format_scatter(ax):
        ax.set_xlabel('vector strength stimulus')
        ax.set_ylabel('vector strength beat')

        ax.set_xlim((0, 1.1))
        ax.set_ylim((0, 1.1))

        ax.tick_params('both', length=3, width=1, which='both')
        ax.text(-0.15, 1, 'A', transform=ax.transAxes, fontweight='bold')

    def format_figure(self):
        sns.despine(self.fig, offset=5, trim=True)
        # self.fig.tight_layout()
        self.gs.tight_layout(self.fig)
        self.fig.subplots_adjust(right=.8)


# ------------------------------------------------------------------------------------------------------
# get all trials with contrast 20%, significant locking to beat or stimulus and |df|>30 to avoid confusion of stimulus
# and EOD
dat = ana.Decoding() * ana.Decoding.Beat() * ana.Decoding.Stimulus() * ana.Runs() & dict(contrast=20) \
      & ['vs_stimulus >= crit_stimulus', 'vs_beat >= crit_beat'] & 'ABS(delta_f) > 30'

df = pd.DataFrame(dat.fetch())
df[r'$\nu$(stimulus) - $\nu$(AM)'] = df.vs_stimulus - df.vs_beat
df['beat/EODf'] = df.beat / df.eod

with Figure07(filename='figures/figure_locking_beat_vs_stimulus.pdf') as (fig, ax):
    for cell, df_cell in df.groupby('cell_id'):
        dfm = df_cell.groupby(['delta_f']).mean()
        ax['difference'].plot(dfm['beat/EODf'], dfm[r'$\nu$(stimulus) - $\nu$(AM)'], '-', label=cell)

    df2 = df[(np.abs(df.delta_f) <= 200) | (df.vs_beat < df.crit_beat)]
    df2 = df2.groupby(['cell_id', 'delta_f']).mean().reset_index()
    h = ax['scatter'].scatter(df2.vs_stimulus, df2.vs_beat, c=df2['beat/EODf'], cmap=plt.get_cmap('coolwarm'),
                              edgecolors='w', lw=.5)

    df2 = df[(np.abs(df.delta_f) > 200) & (df.vs_beat >= df.crit_beat)]
    df2 = df2.groupby(['cell_id', 'delta_f']).mean().reset_index()
    ax['scatter'].scatter(df2.vs_stimulus, df2.vs_beat, c=df2['beat/EODf'], cmap=plt.get_cmap('coolwarm'),
                          edgecolors='w', lw=.5, marker=(5, 1), s=30)

    ax['scatter'].plot(*2 * (np.linspace(0, 1, 2),), '--k', zorder=-10)
    cb = plt.colorbar(h, ax=ax['scatter'], shrink=.8, pad=0.0, aspect=20)
    cb.set_label(r'$\Delta f$/EOD frequency')
    cb.outline.set_linewidth(0)
    cb.set_ticks(np.arange(-.5, 1, .5))
    ax['color'] = cb.ax
