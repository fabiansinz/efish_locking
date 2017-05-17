from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats

from locking import analyses as alys, colors, colordict
from locking import data
from locking import mkdir
from locking import sanity
from locking.data import Baseline
from scripts.config import params as plot_params, FormatedFigure
import pycircstat as circ


def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.png' % (cell['cell_id'], contrast)


class FigurePyramidals(FormatedFigure):
    def __init__(self, filename=None):
        self.filename = filename

    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 7))
            gs = plt.GridSpec(3, 12)
            self.ax = {
                'spectrum': self.fig.add_subplot(gs[:2, 4:]),
                'ISI': self.fig.add_subplot(gs[0, :4]),
                'cycle': self.fig.add_subplot(gs[1, :4]),
                'vs_freq': self.fig.add_subplot(gs[2, :4]),
            }
            self.ax['cycle_ampl'] = self.ax['cycle'].twinx()
            self.ax['circ'] = self.fig.add_subplot(gs[2, 4:8])
            self.ax['contrast'] = self.fig.add_subplot(gs[2, 8:])

    @staticmethod
    def format_ISI(ax):
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])
        ax.set_xlim((0, 15))
        ax.legend()
        ax.text(-0.2, 1, 'A', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_spectrum(ax):
        ax.set_xlim((0, 1000))
        ax.tick_params('x', length=3, width=1)
        ax.spines['bottom'].set_linewidth(1)
        ax.legend(loc='top left', ncol=2, bbox_transform=ax.transAxes)
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        ax.set_yticks([])
        ax.set_xlabel('frequency [Hz]')
        ax.text(-0.05, 1, 'C', transform=ax.transAxes, fontweight='bold')

    # @staticmethod
    # def format_cycle(ax):
    #     sns.despine(ax=ax, left=True)
    #     ax.legend(loc='upper right')
    #     ax.text(-0.2, 1, 'B', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_cycle(ax):
        ax.text(-0.16, 1.01, 'B', transform=ax.transAxes, fontweight='bold')
        yl, yh = ax.get_ylim()
        ax.set_ylim((yl, 1.6 * yh))
        ax.set_yticks([])
        ax.set_xticks(np.linspace(-.6, 1.2, 7))
        ax.set_xticklabels(np.linspace(-.6, 1.2, 7))
        # ax.tick_params(axis='both', length=0, width=0, which='major')
        ax.set_xlim((-.6, 1.2))
        ax.set_xlabel('time [ms]')
        sns.despine(ax=ax, left=True, right=True, trim=True)
        ax.legend(ncol=1)

    @staticmethod
    def format_cycle_ampl(ax):
        ax.tick_params(axis='y', length=3, width=1, which='major')
        ax.set_yticks(np.arange(-2,4))
        sns.despine(ax=ax, trim=True)
        yl, yh = ax.get_ylim()
        ax.yaxis.set_label_position("left")
        ax.set_ylim((yl, 1.6 * yh))
        ax.set_xlim((-.6, 1.2))
        ax.legend(ncol=1, bbox_to_anchor=((.6, 1.)))

    @staticmethod
    def format_vs_freq(ax):
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('vector strength stimulus')
        ax.set_xlim((0, 1800))
        ax.set_ylim((0, 1))
        ax.tick_params('y', length=3, width=1)
        ax.text(-0.2, 1, 'D', transform=ax.transAxes, fontweight='bold')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.set_xticks(np.arange(0,2000,500))
        sns.despine(ax=ax, trim=True)
        ax.legend(bbox_to_anchor=(1,1.1))

    @staticmethod
    def format_circ(ax):
        ax.set_ylim((0, 1))
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_xlabel('circular std')

        ax.set_yticks([])
        ax.text(-0.1, 1, 'E', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, left=True, trim=True)
        ax.legend(bbox_to_anchor=(1,1.1))

    @staticmethod
    def format_contrast(ax):
        ax.set_ylim((0, 1.0))
        ax.set_xlabel('contrast [%]')

        ax.set_ylabel('')
        ax.text(-0.1, 1, 'F', transform=ax.transAxes, fontweight='bold')
        # ax.tick_params('y', length=0, width=0)
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(.1,1.1))

        ax.spines['bottom'].set_linewidth(1)
        sns.despine(ax=ax, trim=True, left=True)

    def format_figure(self):
        fig.tight_layout()
        fig.subplots_adjust(left=0.075, right=0.95)


if __name__ == "__main__":
    f_max = 2000  # Hz
    contrast = 20
    restr = dict(cell_id='2014-11-26-ad', contrast=contrast, am=0, n_harmonics=0, refined=True)

    line_colors = colors
    # target_trials = alys.FirstOrderSpikeSpectra() * data.Runs() & restr
    target_trials = alys.FirstOrderSpikeSpectra() * data.Runs() & restr

    with FigurePyramidals(filename='figures/figure08factor-pyramidals.pdf') as (fig, ax):
        # --- plot ISI histogram
        data.ISIHistograms().plot(ax=ax['ISI'], restrictions=restr)


        if Baseline.SpikeTimes() & restr:
            times = (Baseline.SpikeTimes() & restr).fetch1['times'] / 1000
            eod, sampling_rate = (Baseline() & restr).fetch1['eod', 'samplingrate']
            period = 1 / eod
            t = (times % period)
            nu = circ.vector_strength(t / period * 2 * np.pi)
            print('Vector strength', nu)
            print('p-value', np.exp(-len(times) * nu ** 2))
            # Baseline().plot_psth(ax['cycle'], restr)
        # --- plot baseline psths
        if data.BaseRate() & restr:
            (data.BaseRate() & restr).plot(ax['cycle'], ax['cycle_ampl'], find_range=False)
        #====================================================================================
        # --- plot spectra
        y = [0]
        stim_freq, eod_freq, deltaf_freq = [], [], []
        freq_log = []
        for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
            print(u"\t\t\u0394 f=%.2f" % spec['delta_f'])

            if i == 0:
                ax['spectrum'].plot([20, 20], [12.5, 13], '-', color='darkslategray', lw=2,
                                    solid_capstyle='butt')
                ax['spectrum'].text(40, 12.65, '0.5 vector strength', fontsize=6)


            f, v = spec['frequencies'], spec['vector_strengths']
            if spec['delta_f'] in freq_log:
                continue
            else:
                freq_log.append(spec['delta_f'])
            idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            ax['spectrum'].fill_between(f[idx], y[-1] + 0 * f[idx], y[-1] + v[idx], lw=0, color='darkslategray')

            y.append(y[-1] + .8)
            stim_freq.append(spec['eod'] + spec['delta_f'])
            deltaf_freq.append(spec['delta_f'])
            eod_freq.append(spec['eod'])

        ax['spectrum'].plot(eod_freq, y[:-1], '--', zorder=-1, lw=2, dashes=(3, 7), color=colordict['eod'], label='EOD')
        ax['spectrum'].plot(stim_freq, y[:-1], '-', zorder=-1, lw=2, dashes=(3, 7), color=colordict['stimulus'],
                            label='stimulus')
        # ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-', zorder=-1, lw=1, color=line_colors[2],
        #                     label=r'$|\Delta f|$')

        #====================================================================================
        rel_pu = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() * data.Cells() \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, cell_type='p-unit', am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \

        df_pu = pd.DataFrame(rel_pu.fetch())
        df_pu['spread'] = df_pu['stim_std'] / df_pu['eod'] / 2 / np.pi
        df_pu['jitter'] = df_pu['stim_std']  # rename to avoid conflict with std function
        df_pu['cell type'] = 'p-units'
        #====================================================================================
        # exclude runs that have only one spike and, thus, artificially high locking
        rel_py = data.Runs() * alys.FirstOrderSignificantPeaks() * alys.StimulusSpikeJitter() \
                 * data.Cells() * sanity.SpikeCheck.SpikeCount() \
                 & 'spike_count > 1' \
                 & dict(eod_coeff=0, baseline_coeff=0, refined=1, am=0, n_harmonics=0) \
                 & 'stimulus_coeff = 1' \
                 & 'frequency > 0' \
                 & ['cell_type="i-cell"', 'cell_type="e-cell"']
        print('n={0} cells tested'.format(len(data.Cells() & ['cell_type="i-cell"', 'cell_type="e-cell"'])))
        print('n={0} cells locking'.format(len(data.Cells() & rel_py)))

        df_py = pd.DataFrame(rel_py.fetch())
        df_py['spread'] = df_py['stim_std'] / df_py['eod'] / 2 / np.pi
        df_py['jitter'] = df_py['stim_std']  # rename to avoid conflict with std function
        df_py['cell type'] = 'pyramidal'
        #====================================================================================
        df = pd.concat([df_pu[df_pu.stimulus_coeff == 1], df_py[df_py.stimulus_coeff == 1]])
        for (c, ct), dat in df.groupby(['cell_id', 'cell type']):
            mu = dat.groupby('contrast').mean().reset_index()
            s = dat.groupby('contrast').std().reset_index()
            pp = sns.pointplot('contrast', 'vector_strength', data=dat, ax=ax['contrast'],
                          palette={'p-units': sns.xkcd_rgb['azure'],
                                   'pyramidal': sns.xkcd_rgb['dark fuchsia']},
                          order=[10, 20], hue='cell type', alpha=1, scale=.5)

        #====================================================================================
        print(r"contrast: \rho={0}    p={1}".format(*stats.pearsonr(df_py.contrast, df_py.vector_strength)))

        df_py = df_py[df_py.contrast == 20]
        df_pu = df_pu[df_pu.contrast == 20]

        print(
            r'Correlation stimulus frequency and locking \rho={:.2g}, p={:.2g}'.format(
                *stats.pearsonr(df_py.eod + df_py.delta_f,
                               df_py.vector_strength)))
        print(r'Correlation jitter and locking \rho={:.2g}, p={:.2g}' \
              .format(*stats.pearsonr(df_py.jitter, df_py.vector_strength)))
        # print(r'Correlation spread and locking \rho=%.2g, p=%.2g' % \
        #       stats.pearsonr(df_py.spread, df_py.vector_strength))

        #====================================================================================


        ax['vs_freq'].scatter(df_pu.frequency, df_pu.vector_strength, edgecolors='w', lw=.5,
                              color=sns.xkcd_rgb['azure'], \
                              label='p-units')
        ax['vs_freq'].scatter(df_py.frequency, df_py.vector_strength, edgecolors='w', lw=.5,
                              color=sns.xkcd_rgb['dark fuchsia'], \
                              label='pyramidal')

        # --- circular variance scatter plots
        ax['circ'].scatter(df_pu.jitter, df_pu.vector_strength, edgecolors='w', lw=.5,
                           color=sns.xkcd_rgb['azure'], \
                           label='p-units'
                           )
        ax['circ'].scatter(df_py.jitter, df_py.vector_strength, edgecolors='w', lw=.5,
                           color=sns.xkcd_rgb['dark fuchsia'], \
                           label='pyramidal'
                           )


