import itertools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from locking import mkdir
from locking.analyses import SecondOrderSpikeSpectra
from locking.data import *
from locking.modelling import *
from scripts.plot_settings import params as plot_params


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
                self.ax['sim_isi'] = inset_axes(self.ax['sim_spike_spectrum'], width=.6, height=.6, loc=1 ,
                                                bbox_to_anchor=(1.15, 1.1),
                                                bbox_transform=self.ax['sim_spike_spectrum'].transAxes
                                                )
                self.ax['real_isi'] = inset_axes(self.ax['real_spike_spectrum'], width=.6, height=.6, loc=1,
                                                 bbox_to_anchor=(1.15, 1.1),
                                                 bbox_transform=self.ax['real_spike_spectrum'].transAxes
                                                 )

        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        fig, ax = self.fig, self.ax

        ax['real_spike_spectrum'].set_ylim((0, 1.5))
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
        ax['real_spike_spectrum'].legend_.set_bbox_to_anchor((1.25, .4),
                                                             transform=ax['real_spike_spectrum'].transAxes)
        ax['stimulus_spectrum'].legend_.set_bbox_to_anchor((1.2, 1))
        sns.despine(ax=self.ax['real_isi'], left=True, trim=True)
        sns.despine(ax=self.ax['sim_isi'], left=True, trim=True)

        self.ax['real_isi'].set_yticks([])
        self.ax['sim_isi'].set_yticks([])
        self.ax['real_isi'].set_xticks(range(0,20,5))
        # self.ax['sim_isi'].set_xticks(range(0,20,5))

        ax['stimulus_spectrum'].text(-0.1, 1, 'A', transform=ax['stimulus_spectrum'].transAxes, fontweight='bold')
        ax['membrane_spectrum'].text(-0.1, 1, 'B', transform=ax['membrane_spectrum'].transAxes, fontweight='bold')
        ax['sim_spike_spectrum'].text(-0.1, 1, 'C', transform=ax['sim_spike_spectrum'].transAxes, fontweight='bold')
        ax['real_spike_spectrum'].text(-0.1, 1, 'D', transform=ax['real_spike_spectrum'].transAxes, fontweight='bold')

        ax['real_spike_spectrum'].set_xticklabels(ax['real_spike_spectrum'].get_xticks())
        for aname in ['stimulus_spectrum', 'membrane_spectrum', 'sim_spike_spectrum']:
            for tk in  ax[aname].get_xticklabels():
                tk.set_visible(False)

        if self.filename is not None:
            self.fig.savefig(self.filename)

        plt.close(fig)

    def __call__(self, *args, **kwargs):
        return self

for ri, hs in itertools.product([5,15],[0,1]):
    restrictions = dict(
        id='nwgimproved',
        cell_id='2014-12-03-ai',
        # run_id=13,
        run_id=ri,
        harmonic_stimulation=hs

    )

    for key in (PUnitSimulations() & restrictions).fetch.keys():
        print('Processing', key)
        dir = 'figures/figure04/' + key['id']
        mkdir(dir)
        df = (Runs() & key).fetch1['delta_f']
        with Figure04(filename='{dir}/figure04_{cell_id}_{df}_harmonics{harmonic_stimulation}.png'.format(dir=dir, df=df, **key)) as (fig, ax):
            PUnitSimulations().plot_stimulus_spectrum(key, ax['stimulus_spectrum'])
            PUnitSimulations().plot_membrane_potential_spectrum(key, ax['membrane_spectrum'])
            PUnitSimulations().plot_spike_spectrum(key, ax['sim_spike_spectrum'])

            restrictions = dict(key, refined=True)
            SecondOrderSpikeSpectra().plot(ax['real_spike_spectrum'], restrictions, f_max=2000)
            ISIHistograms().plot(ax['real_isi'], restrictions)
            PUnitSimulations().plot_isi(key, ax['sim_isi'])
