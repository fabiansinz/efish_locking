from plot_settings import params as plot_params
import matplotlib.pyplot as plt

__author__ = 'fabee'
from analyses import *



class Figure01:

    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7,7))
            gs = plt.GridSpec(3,6)
            self.ax = {
                'spectrum': self.fig.add_subplot(gs[1:,:-2]),
                'ISI': self.fig.add_subplot(gs[1,-2:]),
                'FI': self.fig.add_subplot(gs[2,-2:]),
            }
            self.ax['violin'] = self.fig.add_subplot(gs[0,:-1])

        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):

        fig, ax = self.fig, self.ax

        # -- ISI
        sns.despine(ax=ax['ISI'], left=True)
        ax['ISI'].set_yticks([])

        # --- FI
        sns.despine(ax=ax['FI'])


        # --- phase locking histograms
        ax['violin'].set_ylim((0,2*np.pi))
        ax['violin'].set_yticks(np.linspace(0,2*np.pi,5))
        ax['violin'].set_yticklabels([r'$0$',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{4}$', r'$2\pi$'])

        sns.despine(ax=ax['violin'], trim=True)

        # # --- eod and stimulus locking
        # for what in ['stimulus', 'eod']:
        #     sns.despine(ax=ax[what], left=True)
        #     ax[what].set_yticks([])
        #     ax[what].set_ylim((0,1.2))
        #     ax[what].set_xticks(np.linspace(0,2*np.pi,5))
        #     ax[what].set_xticklabels([r'$0$',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{4}$', r'$2\pi$'])

        # -- spectrum
        ax['spectrum'].set_xlim((0,2400))
        ax['spectrum'].legend(loc='best')
        sns.despine(ax=ax['spectrum'], left=True, trim=True, offset=5)
        ax['spectrum'].set_yticks([])
        ax['spectrum'].set_xlabel('frequency [Hz]')
        fig.tight_layout()
        plt.show()
        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self

class MultiContrastFigure:

    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        self.fig, self.ax = plt.subplots(facecolor='w')
        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):

        fig, ax = self.fig, self.ax
        ax.set_xlim((0,2400))
        ax.legend(loc='best')
        # ax.set_ylim((0,1))
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        ax.set_yticks([])
        ax.set_xlabel('frequency [Hz]')
        fig.tight_layout()
        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self
