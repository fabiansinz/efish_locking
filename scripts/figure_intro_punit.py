from seaborn.utils import _set_spine_position

from locking import mkdir
from locking.data import ISIHistograms, FICurves, BaseRate
from scripts.config import params as plot_params, FormatedFigure
# mpl.use('Agg')      # With this line = figure disappears; without this line = warning
import matplotlib.pyplot as plt
from locking.analyses import *
import pycircstat as circ

def generate_filename(cell, contrast):
    dir = 'figures/figure_intro/{cell_type}/'.format(**cell)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


class FigureIntroPunit(FormatedFigure):
    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 7))
            gs = plt.GridSpec(3, 2)
            self.ax = {
                'scatter': self.fig.add_subplot(gs[-1, :]),
                # 'spectrum': self.fig.add_subplot(gs[1:, :-1]),
                'ISI': self.fig.add_subplot(gs[1, 1]),
                'EOD': self.fig.add_subplot(gs[1, 0]),

            }
            self.ax['scatter_base'] =  self.fig.add_subplot(gs[0, :])
            self.ax['EOD_ampl'] = self.ax['EOD'].twinx()
        self.gs = gs

    @staticmethod
    def format_ISI(ax):
        sns.despine(ax=ax, left=True)
        ax.set_yticks([])
        ax.legend()
        ax.text(-0.1, 1.01, 'C', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_scatter_base(ax):
        sns.despine(ax=ax, left=True)
        ax.set_yticks([])
        ax.legend(ncol=3, bbox_to_anchor=(1,1.1), loc=1)
        ax.text(-.07, 1.01, 'A', transform=ax.transAxes, fontweight='bold')


    @staticmethod
    def format_EOD(ax):

        ax.text(-0.16, 1.01, 'B', transform=ax.transAxes, fontweight='bold')
        yl, yh = ax.get_ylim()
        ax.set_ylim((yl, 1.2 * yh))
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0,1.2,5))
        ax.set_xticklabels(np.linspace(0,1.2,5))
        # ax.tick_params(axis='both', length=0, width=0, which='major')
        ax.set_xlim((0, 1.2 ))
        ax.set_xlabel('time [ms]')
        sns.despine(ax=ax, left=True, right=True, trim=True)
        ax.legend(ncol=1)

    @staticmethod
    def format_EOD_ampl(ax):
        ax.tick_params(axis='y', length=3, width=1, which='major')
        sns.despine(ax=ax, trim=True)
        yl, yh = ax.get_ylim()
        ax.yaxis.set_label_position("left")
        ax.set_ylim((yl, 1.3*yh))
        ax.set_xlim((0, 1.2 ))

        # ax.legend(ncol=1, bbox_to_anchor=((.3,1.)))
        ax.legend(ncol=1, bbox_to_anchor=((.6,1.)))



    @staticmethod
    def format_scatter(ax):
        sns.despine(ax=ax, left=True)
        ax.set_xlabel('time [EOD cycles]')
        ax.legend(ncol=4, loc=0, bbox_to_anchor=(1,1.1))
        ax.text(-0.07, 1.02, 'D', transform=ax.transAxes, fontweight='bold')

    def format_figure(self):
        # self.fig.tight_layout()
        self.fig.subplots_adjust(left=.1, top=0.95, hspace=.3)
        # self.gs.tight_layout(self.fig)


if __name__ == "__main__":

    runs = Runs()
    for cell in (Cells() & dict(cell_type='p-unit', cell_id='2014-12-03-ao')).fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])
        contrast = 20

        target_trials = runs & cell & dict(contrast=contrast, am=0, n_harmonics=0, delta_f=200)

        if len(target_trials) > 0:
            with FigureIntroPunit(filename=generate_filename(cell, contrast=contrast)) as (fig, ax):
                # --- plot baseline spikes
                if Baseline() & cell:
                    (Baseline() & cell).plot_raster(ax['scatter_base'])

                # --- plot baseline psths
                if BaseRate() & cell:
                    (BaseRate() & cell).plot(ax['EOD'], ax['EOD_ampl'])

                if Baseline.SpikeTimes() & cell:
                    times = (Baseline.SpikeTimes() & cell).fetch1['times'] / 1000
                    eod, sampling_rate = (Baseline() & cell).fetch1['eod', 'samplingrate']
                    period = 1 / eod
                    t = (times % period)
                    nu = circ.vector_strength(t / period * 2 * np.pi)
                    print('Vector strength', nu)
                    print('p-value', np.exp(-len(times) * nu ** 2))

                # --- plot ISI histogram
                ISIHistograms().plot(ax=ax['ISI'], restrictions=cell)


                EODStimulusPSTSpikes().plot_single(ax=ax['scatter'], restrictions=target_trials)
