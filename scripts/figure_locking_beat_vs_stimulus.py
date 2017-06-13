from scipy.interpolate import interp1d

from locking import analyses as ana, colordict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from locking import mkdir
from scripts.config import params as plot_params, FormatedFigure
from scipy import stats
from scipy import interp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datajoint as dj


class FigureBeatStim(FormatedFigure):
    def prepare(self):
        sns.set_context('paper')
        sns.set_style('ticks')
        sns.set_palette('PuBuGn_d', n_colors=len(pd.unique(df.cell_id)))
        with plt.rc_context(plot_params):
            self.ax = {}
            self.fig = plt.figure(figsize=(7, 2.2), dpi=400)

            self.ax['difference'] = self.fig.add_subplot(1, 3, 3)
            self.ax['scatter'] = self.fig.add_subplot(1, 3, 1)
            self.ax['scatter2'] = self.fig.add_subplot(1, 3, 2)

    @staticmethod
    def format_difference(ax):
        # ax.legend(bbox_to_anchor=(1.6, 1.05), bbox_transform=ax.transAxes, prop={'size': 6})
        ax.set_xlabel(r'$\Delta f/$EODf')
        ax.set_ylabel(r'$\nu$(stimulus) - $\nu$($\Delta f$)')
        ax.set_xlim((-.6, .6))
        ax.set_xticks(np.arange(-.5, 1, .5))
        ax.tick_params('both', length=3, width=1, which='both')
        ax.set_ylim((-.8, 0.5))
        ax.set_yticks(np.arange(-.75, .75, .25))
        ax.text(-0.4, 1, 'C', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_colorscatter(ax):
        ax.tick_params('y', length=0, width=0, which='both', pad=-.15)

    @staticmethod
    def format_colorscatter2(ax):
        ax.tick_params('y', length=0, width=0, which='both', pad=-.15)

    @staticmethod
    def format_scatter(ax):
        ax.set_ylabel(r'$\nu$(stimulus)')
        ax.set_xlabel(r'$\nu$($\Delta f$)')

        ax.set_xlim((0, 1.1))
        ax.set_ylim((0, 1.1))
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])

        ax.tick_params('both', length=3, width=1, which='both')
        ax.text(-0.3, 1, 'A', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_scatter2(ax):
        ax.set_ylabel(r'$\nu$(stimulus)')
        ax.set_xlabel(r'$\nu$($\Delta f$)')

        ax.set_xlim((0, 1.1))
        ax.set_ylim((0, 1.1))
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])

        ax.tick_params('both', length=3, width=1, which='both')
        ax.text(-0.3, 1, 'B', transform=ax.transAxes, fontweight='bold')

    def format_figure(self):
        sns.despine(self.fig, offset=1, trim=True)
        self.fig.tight_layout()
        # self.fig.subplots_adjust(right=.8)


def plot_locking(df, ax, legend=False):
    n = []
    s = 20
    idx = (df.vs_beat >= df.crit_beat) & (df.vs_stimulus < df.crit_stimulus)
    n.append(idx.sum())
    df2 = df[idx].groupby(['cell_id', 'delta_f']).mean().reset_index()
    ax.scatter(df2.vs_beat, df2.vs_stimulus, color=colordict['delta_f'],
               edgecolors='w', lw=.1, s=s, label=r'$\Delta f$ only')

    idx = (df.vs_beat < df.crit_beat) & (df.vs_stimulus >= df.crit_stimulus)
    n.append(idx.sum())
    df2 = df[idx].groupby(['cell_id', 'delta_f']).mean().reset_index()
    ax.scatter(df2.vs_beat, df2.vs_stimulus, color=colordict['stimulus'],
               edgecolors='w', lw=.1, s=s, label='stimulus only')

    idx = (df.vs_beat >= df.crit_beat) & (df.vs_stimulus >= df.crit_stimulus)
    n.append(idx.sum())
    df2 = df[idx].groupby(['cell_id', 'delta_f']).mean().reset_index()
    ax.scatter(df2.vs_beat, df2.vs_stimulus, color=sns.xkcd_rgb['teal blue'],
               edgecolors='w', lw=.1, s=s, label='both')

    ax.set_aspect(1.)

    axins = inset_axes(ax,
                       width="20%",  # width = 30% of parent_bbox
                       height="20%",  # height : 1 inch
                       loc=4)
    axins.bar(0, n[0], color=colordict['delta_f'], align='center')
    axins.bar(1, n[1], color=colordict['stimulus'], align='center')
    axins.bar(2, n[2], color=sns.xkcd_rgb['teal blue'], align='center')
    axins.axis('off')
    ax.plot(*2 * (np.linspace(0, 1, 2),), '--k', zorder=-10)
    print(n)
    if legend:
        ax.legend(ncol=1, prop={'size': 6}, bbox_to_anchor=(.6, 1.1))


# ------------------------------------------------------------------------------------------------------
# get all trials with contrast 20%, significant locking to beat or stimulus and |df|>30 to avoid confusion of stimulus
# and EOD
dat = ana.Decoding() * ana.Cells() * ana.Decoding.Beat() * ana.Decoding.Stimulus() * ana.Runs() \
      & dict(contrast=20, am=0) \
      & ['vs_stimulus >= crit_stimulus', 'vs_beat >= crit_beat'] & 'ABS(delta_f) > 30'
df = pd.DataFrame(dat.fetch())
df[r'$\nu$(stimulus) - $\nu$($\Delta f$)'] = df.vs_stimulus - df.vs_beat
df['beat/EODf'] = df.beat / df.eod

t = np.linspace(-.6, .6, 50)
with FigureBeatStim(filename='figures/figure07beat-vs-stimulus.pdf') as (fig, ax):
    interps = []
    for cell, df_cell in df.groupby('cell_id'):
        dfm = df_cell.groupby(['delta_f']).mean()
        if len(dfm) > 1:
            f = interp1d(dfm['beat/EODf'], dfm[r'$\nu$(stimulus) - $\nu$($\Delta f$)'], fill_value=np.nan,
                         bounds_error=False)
            interps.append(f(t))
            ax['difference'].plot(dfm['beat/EODf'], dfm[r'$\nu$(stimulus) - $\nu$($\Delta f$)'], '-', lw=1, label=cell,
                                  color='lightgrey')
    ax['difference'].plot(t, np.nanmean(interps, axis=0), '-k', lw=1)
    ax['difference'].plot(t, 0 * t, '--', color='k', lw=1)

    plot_locking(df, ax['scatter'], legend=True)
    plot_locking(df[np.abs(df.delta_f) > 200], ax['scatter2'], legend=True)
