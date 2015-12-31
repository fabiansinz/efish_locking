from analyses import SecondOrderSpikeSpectra
from modelling import *
import matplotlib.pyplot as plt
from plot_settings import params as plot_params


class Figure04:
    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        with plt.rc_context(plot_params):
            self.fig, self.ax = plt.subplots(4, 1, figsize=(7, 7), dpi=400, sharex=True)
            self.ax = {
                'stimulus_spectrum': self.ax[0],
                'membrane_spectrum': self.ax[1],
                'sim_spike_spectrum': self.ax[2],
                'real_spike_spectrum': self.ax[2],
            }

        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        fig, ax = self.fig, self.ax


        sns.despine(ax=ax['stimulus_spectrum'], left=True, offset=0)
        ax['stimulus_spectrum'].set_xticklabels([])
        sns.despine(ax=ax['membrane_spectrum'], left=True, offset=0)
        sns.despine(ax=ax['sim_spike_spectrum'], offset=0)

        for a in ax.values():
            a.tick_params('both', length=3, width=1, which='both')


        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self


for key in PUnitSimulations().project().fetch.as_dict:
    with Figure04(filename='figure04_' + key['cell_id']) as (fig, ax):
        PUnitSimulations().plot_stimulus_spectrum(key, ax['stimulus_spectrum'])
        PUnitSimulations().plot_membrane_potential_spectrum(key, ax['membrane_spectrum'])
        PUnitSimulations().plot_spike_spectrum(key, ax['sim_spike_spectrum'])
        #----------------------------------
        # TODO: Remove this later
        from IPython import embed
        embed()
        exit()
        #----------------------------------

        SecondOrderSpikeSpectra().plot(ax['ispectrum'],extrac_restr,f_max)
    plt.show()
    exit()
