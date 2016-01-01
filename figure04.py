from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from analyses import SecondOrderSpikeSpectra
from helpers import mkdir
from modelling import *
from schemata import *
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
                'real_spike_spectrum': self.ax[3],
            }
            with sns.axes_style('ticks'):
                self.ax['real_isi'] = inset_axes(self.ax['real_spike_spectrum'], width= .6, height=.6, loc=1)
                self.ax['sim_isi'] = inset_axes(self.ax['sim_spike_spectrum'], width= .6, height=.6, loc=1)


        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        fig, ax = self.fig, self.ax

        ax['real_spike_spectrum'].set_ylim((0,1.5))
        sns.despine(ax=ax['stimulus_spectrum'], left=True, offset=0)
        ax['stimulus_spectrum'].set_xticklabels([])
        sns.despine(ax=ax['membrane_spectrum'], left=True, offset=0)
        sns.despine(ax=ax['sim_spike_spectrum'], offset=0)
        sns.despine(ax=ax['real_spike_spectrum'], offset=0, trim=True)

        ax['real_spike_spectrum'].set_xlabel('frequency [Hz]')
        ax['real_spike_spectrum'].set_yticks([0, .5, 1])


        for a in ax.values():
            a.tick_params('both', length=3, width=1, which='both')


        fig.tight_layout()
        fig.subplots_adjust(right=0.80)
        ax['real_spike_spectrum'].legend_.set_bbox_to_anchor((1.25,1))
        ax['stimulus_spectrum'].legend_.set_bbox_to_anchor((1.2,1))
        sns.despine(ax = self.ax['real_isi'], left=True, trim=True)
        sns.despine(ax = self.ax['sim_isi'], left=True, trim=True)

        self.ax['real_isi'].set_yticks([])
        self.ax['sim_isi'].set_yticks([])

        ax['stimulus_spectrum'].text(-0.1, 1, 'A', transform=ax['stimulus_spectrum'].transAxes, fontweight='bold')
        ax['membrane_spectrum'].text(-0.1, 1, 'B', transform=ax['membrane_spectrum'].transAxes, fontweight='bold')
        ax['sim_spike_spectrum'].text(-0.1, 1, 'C', transform=ax['sim_spike_spectrum'].transAxes, fontweight='bold')
        ax['real_spike_spectrum'].text(-0.1, 1, 'D', transform=ax['real_spike_spectrum'].transAxes, fontweight='bold')


        if self.filename is not None:
            self.fig.savefig(self.filename)

        plt.close(fig)
    def __call__(self, *args, **kwargs):
        return self


for key in PUnitSimulations().project().fetch.as_dict:
    dir = 'figures/figure04/' + key['id']
    mkdir(dir)
    with Figure04(filename=dir + '/figure04_' + key['cell_id']+  '.pdf') as (fig, ax):
        PUnitSimulations().plot_stimulus_spectrum(key, ax['stimulus_spectrum'])
        PUnitSimulations().plot_membrane_potential_spectrum(key, ax['membrane_spectrum'])
        PUnitSimulations().plot_spike_spectrum(key, ax['sim_spike_spectrum'])

        restrictions = dict(key, refined=True)
        SecondOrderSpikeSpectra().plot(ax['real_spike_spectrum'], restrictions, f_max=2000)
        ISIHistograms().plot(ax['real_isi'], restrictions)
        PUnitSimulations().plot_isi(key, ax['sim_isi'])
