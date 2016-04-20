from locking import mkdir
from locking.data import ISIHistograms, FICurves
from plot_settings import params as plot_params, FormatedFigure
# mpl.use('Agg')      # With this line = figure disappears; without this line = warning
import matplotlib.pyplot as plt
from locking.analyses import *


def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'],)
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


class Figure01(FormatedFigure):
    def prepare(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 7))
            gs = plt.GridSpec(3, 3)
            self.ax = {
                'ispectrum': self.fig.add_subplot(gs[2, :]),
                'scatter': self.fig.add_subplot(gs[1, :]),
                # 'spectrum': self.fig.add_subplot(gs[1:, :-1]),
                'ISI': self.fig.add_subplot(gs[0, 0]),
                'EOD': self.fig.add_subplot(gs[0, 1]),
                'FI': self.fig.add_subplot(gs[0, 2]),
            }
            # self.ax['violin'] = self.fig.add_subplot(gs[1:, -1])
        self.gs = gs

    @staticmethod
    def format_ISI(ax):
        sns.despine(ax=ax, left=True)
        ax.set_yticks([])
        ax.text(-0.1, 1.01, 'A', transform=ax.transAxes, fontweight='bold')

    # @staticmethod
    # def format_spectrum(ax):
    #     ax.set_xlim((0,2400))
    #     ax.legend(bbox_to_anchor=(1.1, 1),  bbox_transform=ax.transAxes)
    #     sns.despine(ax=ax, left=True, trim=True, offset=5)
    #     ax.set_yticks([])
    #     ax.set_xlabel('frequency [Hz]')
    #     ax.text(-0.01, 1.01, 'C', transform=ax.transAxes, fontweight='bold', va='top', ha='right')
    #
    # @staticmethod
    # def format_violin(ax):
    #     ax.set_xlim((0,2*np.pi))
    #     ax.set_xticks(np.linspace(0,2*np.pi,5))
    #     ax.set_xticklabels([r'$0$',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{4}$', r'$2\pi$'])
    #     ax.set_ylabel(r'$\Delta f$ [Hz]')
    #     ax.set_xlabel('Phase')
    #     ax.text(-0.15, 1.01, 'D', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    #     sns.despine(ax=ax, trim=True)

    @staticmethod
    def format_FI(ax):
        sns.despine(ax=ax)
        ax.text(-0.1, 1.01, 'C', transform=ax.transAxes, fontweight='bold')
        ax.legend(loc='upper right')

    @staticmethod
    def format_EOD(ax):
        sns.despine(ax=ax, left=True, trim=True)
        ax.text(-0.1, 1.01, 'B', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_ispectrum(ax):
        sns.despine(ax=ax, trim=True)
        ax.text(-0.1, 1.01, 'D', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_scatter(ax):
        sns.despine(ax=ax)
        ax.set_xlabel('time [EOD cycles]')
        ax.text(-0.1, 1.01, 'E', transform=ax.transAxes, fontweight='bold')

    def format_figure(self):
        self.fig.tight_layout()
        self.gs.tight_layout(self.fig)


if __name__ == "__main__":
    f_max = 2000  # Hz
    delta_f = 200

    # TODO axis in b
    #      xaxis in A
    runs = Runs()
    for cell in (Cells() & dict(cell_type='p-unit')).fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        # line_colors = sns.color_palette('pastel', n_colors=3)
        for spectrum, speaks, base_name in zip([FirstOrderSpikeSpectra(), SecondOrderSpikeSpectra()],
                                               [FirstOrderSignificantPeaks(), SecondOrderSignificantPeaks()],
                                               ['firstorderspectra', 'secondorderspectra']):
            print('\t',base_name)
            for contrast in [5, 10, 20]:
                print("\t\tcontrast: %.2f%%" % (contrast,))

                target_trials = spectrum * runs & cell & dict(contrast=contrast, am=0, n_harmonics=0)

                if len(target_trials) > 0:
                    with Figure01(filename=generate_filename(cell, contrast=contrast, base=base_name)) as (fig, ax):
                        # --- plot baseline psths
                        if Baseline.SpikeTimes() & cell:
                            Baseline().plot_psth(ax['EOD'], cell)

                        # --- plot ISI histogram
                        ISIHistograms().plot(ax=ax['ISI'], restrictions=cell)

                        # --- plot FICurves
                        FICurves().plot(ax=ax['FI'], restrictions=cell)

                        # # --- plot locking
                        # PhaseLockingHistogram().violin_plot(ax['violin'], restrictions=target_trials,
                        #                                     palette=line_colors)

                        mydf = np.unique(target_trials.fetch['delta_f'])
                        extrac_restr = target_trials * speaks & dict(delta_f=mydf[np.argmin(abs(mydf - delta_f))],
                                                                     refined=1)
                        spectrum.plot(ax['ispectrum'], extrac_restr, f_max)

                        EODStimulusPSTSpikes().plot(ax=ax['scatter'], restrictions=target_trials)

                        # # --- plot spectra
                        # y = [0]
                        # stim_freq, eod_freq, deltaf_freq = [], [], []
                        # for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
                        #     print(u"\t\t\u0394 f=%.2f" % spec['delta_f'])
                        #
                        #     f, v = spec['frequencies'], spec['vector_strengths']
                        #     idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
                        #     ax['spectrum'].fill_between(f[idx], y[-1] + 0 * f[idx], y[-1] + v[idx], lw=0,
                        #                                 color='darkslategray')
                        #
                        #     y.append(y[-1] + .8)
                        #     stim_freq.append(spec['eod'] + spec['delta_f'])
                        #     deltaf_freq.append(spec['delta_f'])
                        #     eod_freq.append(spec['eod'])
                        #
                        # ax['spectrum'].plot(eod_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[0], label='EOD')
                        # ax['spectrum'].plot(stim_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[1],
                        #                     label='stimulus')
                        # ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-', zorder=-1, lw=1, color=line_colors[2],
                        #                     label=r'$|\Delta f|$')
