from locking import mkdir
from locking.data import ISIHistograms, FICurves
from scripts.config import params as plot_params, FormatedFigure
# mpl.use('Agg')      # With this line = figure disappears; without this line = warning
import matplotlib.pyplot as plt
from locking.analyses import *


def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/figure_locking/%s/%s/' % (base, cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


class FigureLocking(FormatedFigure):
    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 7))
            gs = plt.GridSpec(3, 2)
            self.ax = {
                'ispectrum': self.fig.add_subplot(gs[2, :]),
                'scatter': self.fig.add_subplot(gs[:2, :]),
            }
            # self.ax['violin'] = self.fig.add_subplot(gs[1:, -1])
        self.gs = gs

    @staticmethod
    def format_ispectrum(ax):
        sns.despine(ax=ax, trim=True)
        ax.text(-0.06, 1.17, 'B', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_scatter(ax):
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel('time [EOD cycles]')
        ax.legend(bbox_to_anchor=(1, 1.07))
        ax.text(-0.06, 1.02, 'A', transform=ax.transAxes, fontweight='bold')

    def format_figure(self):
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.95, hspace=0.5)
        # self.gs.tight_layout(self.fig)


if __name__ == "__main__":
    f_max = 2000  # Hz
    delta_f = 200

    runs = Runs()
    for cell in (Cells() & dict(cell_type='p-unit', cell_id='2014-12-03-ao')).fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        spectrum = SecondOrderSpikeSpectra()
        speaks = SecondOrderSignificantPeaks()
        base_name = 'secondorderspectra'
        for contrast in [20]:
            print("\t\tcontrast: %.2f%%" % (contrast,))

            target_trials = spectrum * runs & cell & dict(contrast=contrast, am=0, n_harmonics=0)

            if len(target_trials) > 0:
                with FigureLocking(filename=generate_filename(cell, contrast=contrast, base=base_name)) as (fig, ax):
                    mydf = np.unique(target_trials.fetch['delta_f'])
                    mydf.sort()
                    extrac_restr = target_trials * speaks & dict(delta_f=mydf[-1],
                                                                 refined=1)

                    spectrum.plot(ax['ispectrum'], extrac_restr.proj(), f_max)

                    EODStimulusPSTSpikes().plot(ax=ax['scatter'], restrictions=target_trials.proj(), repeats=500)
