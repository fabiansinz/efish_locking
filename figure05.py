from locking import mkdir
from plot_settings import params as plot_params, FormatedFigure
from collections import OrderedDict

import matplotlib.pyplot as plt
from locking import data
from locking import analyses as alys
import pandas as pd
import seaborn as sns
import numpy as np


def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


class Figure05(FormatedFigure):
    def __init__(self, filename=None):
        self.filename = filename

    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 7))
            gs = plt.GridSpec(3, 3)
            self.ax = {
                'spectrum': self.fig.add_subplot(gs[:2, :2]),
                'ISI': self.fig.add_subplot(gs[0, 2]),
                'FI': self.fig.add_subplot(gs[1, 2]),
                'vs_freq': self.fig.add_subplot(gs[2, 0]),
            }
            self.ax['circ'] = self.fig.add_subplot(gs[2, 1])
            self.ax['contrast'] = self.fig.add_subplot(gs[2, 2])

    @staticmethod
    def format_ISI(ax):
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])
        ax.set_xlim((0, 15))
        ax.text(-0.2, 1, 'B', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_spectrum(ax):
        ax.set_xlim((0, 2400))
        ax.tick_params('x', length=3, width=1)
        ax.spines['bottom'].set_linewidth(1)
        ax.legend(bbox_to_anchor=(1.1, 1), bbox_transform=ax.transAxes)
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        ax.set_yticks([])
        ax.set_xlabel('frequency [Hz]')
        ax.text(-.1, 1, 'A', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_FI(ax):
        sns.despine(ax=ax)
        ax.legend(loc='upper right')
        ax.text(-0.2, 1, 'C', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_vs_freq(ax):
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('vector strength')
        ax.set_xlim((0, 2400))
        ax.set_ylim((0, 1))
        ax.tick_params('y', length=3, width=1)
        ax.text(-0.2, 1, 'D', transform=ax.transAxes, fontweight='bold')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.legend()

    @staticmethod
    def format_circ(ax):
        ax.set_ylim((0, 1))
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_xlabel('circular std')

        ax.set_yticks([])
        ax.text(-0.2, 1, 'E', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, left=True, trim=True)
        ax.legend()

    @staticmethod
    def format_contrast(ax):
        ax.set_ylim((0, 1.0))
        ax.set_xlabel('contrast')

        ax.set_ylabel('')
        ax.text(-0.2, 1, 'F', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, trim=True, left=True)

    def format_figure(self):
        fig.tight_layout()
        fig.subplots_adjust(left=0.1)


if __name__ == "__main__":
    f_max = 2000  # Hz
    contrast = 20
    restr = dict(cell_id='2014-11-26-ad', contrast=contrast, am=0, n_harmonics=0)

    line_colors = sns.color_palette('pastel', n_colors=3)

    target_trials = alys.FirstOrderSpikeSpectra() * data.Runs() & restr

    with Figure05(filename='figures/figure05.pdf') as (fig, ax):
        # --- plot ISI histogram
        data.ISIHistograms().plot(ax=ax['ISI'], restrictions=restr)

        # --- plot FICurves
        data.FICurves().plot(ax=ax['FI'], restrictions=restr)

        # --- plot spectra
        y = [0]
        stim_freq, eod_freq, deltaf_freq = [], [], []
        for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
            print(u"\t\t\u0394 f=%.2f" % spec['delta_f'])

            f, v = spec['frequencies'], spec['vector_strengths']
            idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            ax['spectrum'].fill_between(f[idx], y[-1] + 0 * f[idx], y[-1] + v[idx], lw=0, color='darkslategray')

            y.append(y[-1] + .8)
            stim_freq.append(spec['eod'] + spec['delta_f'])
            deltaf_freq.append(spec['delta_f'])
            eod_freq.append(spec['eod'])

        ax['spectrum'].plot(eod_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[0], label='EOD')
        ax['spectrum'].plot(stim_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[1], label='stimulus')
        # ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-', zorder=-1, lw=1, color=line_colors[2],
        #                     label=r'$|\Delta f|$')

        # --- scatter plots
        rel_pu = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() * data.Cells() \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, \
                        cell_type='p-unit', am=0, n_harmonics=0) \
                 & 'stimulus_coeff > 0' \
                 & 'frequency > 0'

        df_pu = pd.DataFrame(rel_pu.fetch())
        df_pu['spread'] = df_pu['stim_std'] / df_pu['eod'] / 2 / np.pi
        df_pu['jitter'] = df_pu['stim_std']  # rename to avoid conflict with std function

        rel_py = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() * data.Cells() \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, \
                        am=0, n_harmonics=0) \
                 & 'stimulus_coeff > 0' \
                 & 'frequency > 0' \
                 & ['cell_type="i-cell"', 'cell_type="e-cell"']
        df_py = pd.DataFrame(rel_py.fetch())
        df_py['spread'] = df_py['stim_std'] / df_py['eod'] / 2 / np.pi
        df_py['jitter'] = df_py['stim_std']  # rename to avoid conflict with std function

        print('Correlation stimulus frequency and locking',
              np.corrcoef(df_py.eod + df_py.delta_f, df_py.vector_strength)[1, 0])
        print('Correlation jitter and locking',
              np.corrcoef(df_py.jitter, df_py.vector_strength)[1, 0])
        print('Correlation spread and locking',
              np.corrcoef(df_py.spread, df_py.vector_strength)[1, 0])
        ax['vs_freq'].scatter(df_pu.frequency, df_pu.vector_strength, edgecolors='w', lw=.5, color='steelblue', \
                              label='p-units')
        ax['vs_freq'].scatter(df_py.frequency, df_py.vector_strength, edgecolors='w', lw=.5, color='orangered', \
                              label='pyramidal')

        # --- circular variance scatter plots
        ax['circ'].scatter(df_pu.jitter, df_pu.vector_strength, edgecolors='w', lw=.5, color='steelblue', \
                           label='p-units'
                           )
        ax['circ'].scatter(df_py.jitter, df_py.vector_strength, edgecolors='w', lw=.5, color='orangered', \
                           label='pyramidal'
                           )

        # --- contrast
        df_pu['cell type'] = 'p-units'
        df_py['cell type'] = 'pyramidal'
        df = pd.concat([df_pu[df_pu.stimulus_coeff == 1], df_py[df_py.stimulus_coeff == 1]])

        for (c, ct), dat in df.groupby(['cell_id', 'cell type']):
            mu = dat.groupby('contrast').mean().reset_index()
            s = dat.groupby('contrast').std().reset_index()
            sns.pointplot('contrast', 'vector_strength', data=dat, ax=ax['contrast'],
                          palette={'p-units': 'steelblue', 'pyramidal': 'orangered'},
                          order=[2.5, 5, 10, 20], hue='cell type', alpha=1, scale=.5)
