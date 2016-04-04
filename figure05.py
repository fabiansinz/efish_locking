from locking import mkdir
from plot_settings import params as plot_params
# mpl.use('Agg')      # With this line = figure disappears; without this line = warning
import matplotlib.pyplot as plt
from locking import data, analyses
import seaborn as sns
import numpy as np
def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)


class Figure05:

    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7,7))
            gs = plt.GridSpec(3,3)
            self.ax = {
                'spectrum': self.fig.add_subplot(gs[:2,:2]),
                'ISI': self.fig.add_subplot(gs[0,2]),
                'FI': self.fig.add_subplot(gs[1,2]),
            }


        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):

        fig, ax = self.fig, self.ax

        # -- ISI
        sns.despine(ax=ax['ISI'], left=True)
        ax['ISI'].set_yticks([])
        ax['ISI'].text(-0.01, 1.01, 'A', transform=ax['ISI'].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

        # --- FI
        sns.despine(ax=ax['FI'])
        ax['FI'].text(-0.2, 1.01, 'B', transform=ax['FI'].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        ax['FI'].legend(loc='upper right')



        fig.tight_layout()
        # -- spectrum
        ax['spectrum'].set_xlim((0,2400))
        ax['spectrum'].legend(bbox_to_anchor=(1.1, 1),  bbox_transform=ax['spectrum'].transAxes)
        sns.despine(ax=ax['spectrum'], left=True, trim=True, offset=5)
        ax['spectrum'].set_yticks([])
        ax['spectrum'].set_xlabel('frequency [Hz]')
        ax['spectrum'].text(-0.01, 1.01, 'C', transform=ax['spectrum'].transAxes, fontweight='bold', va='top', ha='right')
        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self


if __name__ == "__main__":
    f_max = 2000 # Hz
    fos = analyses.FirstOrderSpikeSpectra()
    sos = analyses.SecondOrderSpikeSpectra()
    contrast = 20
    restr = dict(cell_id='2014-11-26-ad', contrast=contrast, am=0, n_harmonics = 0)
    runs = data.Runs()

    line_colors = sns.color_palette('pastel', n_colors=3)#sns.xkcd_rgb['golden yellow'], sns.xkcd_rgb['baby blue'], sns.xkcd_rgb['apple green']
    for spectrum, base_name in zip([fos, sos], ['firstorderspectra', 'secondorderspectra']):
        print("contrast: %.2f%%" % (contrast,))

        target_trials = spectrum * runs & restr

        if len(target_trials) > 0:
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
                    ax['spectrum'].fill_between(f[idx], y[-1] + 0*f[idx], y[-1] + v[idx], lw=0, color='darkslategray')

                    y.append(y[-1] + .8)
                    stim_freq.append(spec['eod'] + spec['delta_f'])
                    deltaf_freq.append(spec['delta_f'])
                    eod_freq.append(spec['eod'])

                ax['spectrum'].plot(eod_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[0], label='EOD')
                ax['spectrum'].plot(stim_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[1],label='stimulus')
                ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-',  zorder=-1, lw=1, color=line_colors[2], label=r'$|\Delta f|$')
