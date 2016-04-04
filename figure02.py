import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift

from locking.analyses import *
from locking.data import *
from plot_settings import params as plot_params


def generate_filename(cell, contrast):
    dir = 'figures/figure02/%s/' % (cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)

def gauss(t, m, v):
    return np.exp(-(t-m)**2/2/v)


class Figure02:

    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_context('paper')
        sns.set_style('ticks')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7,7), dpi=400)
            gs = plt.GridSpec(20,4)
            self.ax = {
                'baseline': self.fig.add_subplot(gs[:4,-1]),
                'ispectrum': self.fig.add_subplot(gs[12:,:]),
            }

            with sns.axes_style('whitegrid'):
                self.ax['period'] = self.fig.add_subplot(gs[4:8,-1], polar=True)
            self.ax['psth'] = self.fig.add_subplot(gs[:8,:-1])

            # self.ax['cartoon eod'] = self.fig.add_subplot(gs[8,:3])
            self.ax['cartoon psth'] =  self.fig.add_subplot(gs[8:10,:3])
            # self.ax['cartoon stim'] = self.fig.add_subplot(gs[10,:3])
            self.ax['cartoon psth stim'] =  self.fig.add_subplot(gs[10:12,:3])

            self.ax['spectrum base'] = self.fig.add_subplot(gs[8:10,3:])
            self.ax['spectrum stim'] = self.fig.add_subplot(gs[10:12,3:])
            self.gs = gs
        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):

        fig, ax = self.fig, self.ax

        # psth
        sns.despine(ax=ax['psth'], left=True, trim=True)
        # baseline
        sns.despine(ax=ax['baseline'], left=True, trim=True)

        # --- time
        for k in ['cartoon psth', 'cartoon psth stim']:
            sns.despine(ax=ax[k], left=True, trim=True)
            ax[k].set_yticks([])

        # --- spectrum
        sns.despine(ax=ax['spectrum base'], left=True, trim=True)
        ax['spectrum base'].set_yticks([])

        sns.despine(ax=ax['spectrum stim'], left=True, trim=True)
        ax['spectrum stim'].set_yticks([])

        # ax['period'].set_thetagrids(np.arange(45, 360, 90), frac=1.3)

        sns.despine(ax=ax['ispectrum'], trim=True)

        ax['psth'].text(-0.1, 1, 'A', transform=ax['psth'].transAxes, fontweight='bold')
        ax['baseline'].text(-0.1, 1, 'B', transform=ax['baseline'].transAxes, fontweight='bold')
        ax['period'].text(-0.1, 1, 'C', transform=ax['period'].transAxes, fontweight='bold')
        ax['cartoon psth'].text(-0.1, 1, 'D', transform=ax['cartoon psth'].transAxes, fontweight='bold')
        ax['cartoon psth stim'].text(-0.1, 1, 'E', transform=ax['cartoon psth stim'].transAxes, fontweight='bold')
        ax['ispectrum'].text(-0.1, 1, 'F', transform=ax['ispectrum'].transAxes, fontweight='bold')


        # self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=1.6)
        self.gs.tight_layout(self.fig)
        if self.filename is not None:
            self.fig.savefig(self.filename)
        plt.close(self.fig)

    def __call__(self, *args, **kwargs):
        return self


if __name__ == "__main__":
    f_max = 2000 # Hz
    N = 10
    fos = FirstOrderSpikeSpectra()
    sos = SecondOrderSpikeSpectra()
    delta_f = 200

    runs = Runs()
    # for cell in (Cells() & dict(cell_type='p-unit', cell_id="2014-12-03-ao")).fetch.as_dict:
    for cell in (Cells() & dict(cell_type='p-unit')).fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        line_colors = sns.color_palette('pastel', n_colors=3)#sns.xkcd_rgb['golden yellow'], sns.xkcd_rgb['baby blue'], sns.xkcd_rgb['apple green']
        for contrast in [5, 10, 20]:
            print("contrast: %.2f%%" % (contrast,))

            target_trials = runs & cell & dict(contrast=contrast, am=0, n_harmonics=0)

            if target_trials:
                with Figure02(filename=generate_filename(cell, contrast=contrast)) as (fig, ax):
                    # --- plot ISI histogram
                    EODStimulusPSTSpikes().plot(ax=ax['psth'], restrictions=target_trials)

                    # --- plot baseline psths
                    if Baseline.SpikeTimes() & cell:
                        Baseline().plot_psth(ax['baseline'], cell)

                    # --- plot time cartoon psth baseline
                    eod = target_trials.fetch['eod'].mean()
                    stim_period = 1/(eod-delta_f)
                    var = (1/8/eod)**2
                    t = np.linspace(-N/eod,N/eod, 10000)
                    base = lambda t: np.cos(2*np.pi*eod*t)+1
                    beat = lambda t: np.cos(2*np.pi*delta_f*t)+1
                    stim = lambda t: np.cos(2*np.pi*eod*t)+np.cos(2*np.pi*(eod-delta_f)*t) + 2
                    f_base = sum(gauss(t, mu, var) for mu in np.arange(-N/eod,N/eod,1/eod))
                    f_stim = sum(gauss(t, mu, var)*beat(mu) for mu in np.arange(-N/eod,N/eod,1/eod))

                    ax['period'].fill_between(t/stim_period*2*np.pi, f_stim*0, f_stim,color='deeppink', label='EOD + stimulus')
                    ax['period'].fill_between(t/stim_period*2*np.pi, f_base*0, f_base,color='dodgerblue', label='EOD only')
                    ax['period'].set_yticks([])
                    ax['period'].legend()

                    ax['cartoon psth'].fill_between(t, 0*t, f_base, color='dodgerblue', lw=0)
                    ax['cartoon psth'].plot(t, base(t), '-k')
                    ax['cartoon psth'].set_ylim((0,2.1))

                    ax['cartoon psth stim'].fill_between(t, 0*t, f_stim, color='deeppink', lw=0)
                    ax['cartoon psth stim'].plot(t, stim(t), '-k')
                    ax['cartoon psth stim'].set_ylim((0,4.2))


                    for k in ['cartoon psth', 'cartoon psth stim', 'psth']:
                    # for k in ['cartoon psth', 'cartoon eod', 'cartoon stim', 'cartoon psth stim', 'psth']:

                        ax[k].set_xticks(np.arange(-N/eod,(N+1)/eod, 5/eod))
                        ax[k].set_xlim((-N/eod,N/eod))
                        ax[k].set_xticklabels([])
                    ax['cartoon psth stim'].set_xticklabels(np.arange(-N, N+1, 5))
                    ax['cartoon psth stim'].set_xlabel('time [EOD cycles]')


                    F_base = fftshift(fft(f_base))
                    w_base = fftshift(fftfreq(f_base.size, t[1]-t[0]))
                    idx = abs(w_base) < f_max
                    ax['spectrum base'].plot(w_base[idx], abs(F_base[idx]), '-k')



                    F_stim = fftshift(fft(f_stim))
                    w_stim = fftshift(fftfreq(f_stim.size, t[1]-t[0]))
                    idx = abs(w_stim) < f_max
                    ax['spectrum stim'].plot(w_stim[idx], abs(F_stim[idx]), '-k')

                    for k in ['spectrum base', 'spectrum stim']:
                        ax[k].set_xticks(np.arange(-2*eod, 3*eod, eod))
                        ax[k].set_xlim((-f_max, f_max))
                        ax[k].set_xticklabels([])
                    ax['spectrum stim'].set_xticklabels(np.arange(-2, 3))
                    ax['spectrum stim'].set_xlabel('frequency/EOD')

                    # --- real spectrum
                    mydf = np.unique(target_trials.fetch['delta_f'])

                    extrac_restr = target_trials*SecondOrderSignificantPeaks() & dict(delta_f=mydf[np.argmin(abs(mydf - delta_f))], refined=1)

                    SecondOrderSpikeSpectra().plot(ax['ispectrum'],extrac_restr,f_max)
