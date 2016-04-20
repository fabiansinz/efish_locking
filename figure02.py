import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift

from locking import mkdir
from locking.analyses import *
from locking.data import *
from plot_settings import params as plot_params, FormatedFigure


def generate_filename(cell, contrast):
    dir = 'figures/figure02/%s/' % (cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)

def gauss(t, m, v):
    return np.exp(-(t-m)**2/2/v)


class Figure02(FormatedFigure):
    def prepare(self):
        sns.set_context('paper')
        sns.set_style('ticks')
        with plt.rc_context(plot_params):
            self.fig = plt.figure(figsize=(7, 7), dpi=400)
            gs = plt.GridSpec(5, 4)
            self.ax = {
                'spectrum': self.fig.add_subplot(gs[:3, :3]),
                'violin': self.fig.add_subplot(gs[:3, 3]),
            }

            self.ax['cartoon_psth'] = self.fig.add_subplot(gs[3, :3])
            self.ax['cartoon_psth_stim'] = self.fig.add_subplot(gs[4, :3])

            self.ax['spectrum_base'] = self.fig.add_subplot(gs[3, 3])
            self.ax['spectrum_stim'] = self.fig.add_subplot(gs[4, 3])
        self.gs = gs

    @staticmethod
    def format_spectrum(ax):
        ax.set_xlim((0,2400))
        ax.legend(bbox_to_anchor=(1.1, 1),  bbox_transform=ax.transAxes)
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        ax.set_yticks([])
        ax.set_xlabel('frequency [Hz]')
        ax.text(-0.01, 1.01, 'A', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_violin(ax):
        ax.set_xlim((0,2*np.pi))
        ax.set_xticks(np.linspace(0,2*np.pi,5))
        ax.set_xticklabels([r'$0$',r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{4}$', r'$2\pi$'])
        ax.set_ylabel(r'$\Delta f$ [Hz]')
        ax.set_xlabel('Phase')
        ax.text(-0.15, 1.01, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        sns.despine(ax=ax, trim=True)

    @staticmethod
    def format_cartoon_psth(ax):
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])
        ax.text(-0.01, 1, 'C', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_cartoon_psth_stim(ax):
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])
        ax.text(-0.01, 1, 'D', transform=ax.transAxes, fontweight='bold')

    @staticmethod
    def format_spectrum_base(ax):
        # --- spectrum
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])

    @staticmethod
    def format_spectrum_stim(ax):
        # --- spectrum
        sns.despine(ax=ax, left=True, trim=True)
        ax.set_yticks([])


    def format_figure(self):
        self.gs.tight_layout(self.fig)




if __name__ == "__main__":
    f_max = 2000 # Hz
    N = 10
    delta_f = 200

    runs = Runs()
    # for cell in (Cells() & dict(cell_type='p-unit', cell_id="2014-12-03-ao")).fetch.as_dict:
    for cell in (Cells() & dict(cell_type='p-unit')).fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        line_colors = sns.color_palette('pastel', n_colors=3)
        for contrast in [5, 10, 20]:
            print("contrast: %.2f%%" % (contrast,))

            # target_trials = runs & cell & dict(contrast=contrast, am=0, n_harmonics=0)
            target_trials = SecondOrderSpikeSpectra() * runs & cell & dict(contrast=contrast, am=0, n_harmonics=0)
            if target_trials:
                with Figure02(filename=generate_filename(cell, contrast=contrast)) as (fig, ax):


                    # --- plot spectra
                    y = [0]
                    stim_freq, eod_freq, deltaf_freq = [], [], []
                    for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
                        print(u"\t\t\u0394 f=%.2f" % spec['delta_f'])

                        f, v = spec['frequencies'], spec['vector_strengths']
                        idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
                        ax['spectrum'].fill_between(f[idx], y[-1] + 0 * f[idx], y[-1] + v[idx], lw=0,
                                                    color='darkslategray')

                        y.append(y[-1] + .8)
                        stim_freq.append(spec['eod'] + spec['delta_f'])
                        deltaf_freq.append(spec['delta_f'])
                        eod_freq.append(spec['eod'])

                    ax['spectrum'].plot(eod_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[0], label='EOD')
                    ax['spectrum'].plot(stim_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[1],
                                        label='stimulus')
                    ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-', zorder=-1, lw=1, color=line_colors[2],
                                        label=r'$|\Delta f|$')

                    # --- plot locking
                    PhaseLockingHistogram().violin_plot(ax['violin'], restrictions=target_trials,
                                                        palette=line_colors)


                    # --- plot time cartoon_psth baseline
                    eod = target_trials.fetch['eod'].mean()
                    stim_period = 1/(eod-delta_f)
                    var = (1/8/eod)**2
                    t = np.linspace(-N/eod,N/eod, 10000)
                    base = lambda t: np.cos(2*np.pi*eod*t)+1
                    beat = lambda t: np.cos(2*np.pi*delta_f*t)+1
                    stim = lambda t: np.cos(2*np.pi*eod*t)+np.cos(2*np.pi*(eod-delta_f)*t) + 2
                    f_base = sum(gauss(t, mu, var) for mu in np.arange(-N/eod,N/eod,1/eod))
                    f_stim = sum(gauss(t, mu, var)*beat(mu) for mu in np.arange(-N/eod,N/eod,1/eod))

                    ax['cartoon_psth'].fill_between(t, 0*t, f_base, color='dodgerblue', lw=0)
                    ax['cartoon_psth'].plot(t, base(t), '-k')
                    ax['cartoon_psth'].set_ylim((0,2.1))

                    ax['cartoon_psth_stim'].fill_between(t, 0*t, f_stim, color='deeppink', lw=0)
                    ax['cartoon_psth_stim'].plot(t, stim(t), '-k')
                    ax['cartoon_psth_stim'].set_ylim((0,4.2))


                    for k in ['cartoon_psth', 'cartoon_psth_stim']:
                        ax[k].set_xticks(np.arange(-N/eod,(N+1)/eod, 5/eod))
                        ax[k].set_xlim((-N/eod,N/eod))
                        ax[k].set_xticklabels([])
                    ax['cartoon_psth_stim'].set_xticklabels(np.arange(-N, N+1, 5))
                    ax['cartoon_psth_stim'].set_xlabel('time [EOD cycles]')


                    F_base = fftshift(fft(f_base))
                    w_base = fftshift(fftfreq(f_base.size, t[1]-t[0]))
                    idx = abs(w_base) < f_max
                    ax['spectrum_base'].plot(w_base[idx], abs(F_base[idx]), '-k')



                    F_stim = fftshift(fft(f_stim))
                    w_stim = fftshift(fftfreq(f_stim.size, t[1]-t[0]))
                    idx = abs(w_stim) < f_max
                    ax['spectrum_stim'].plot(w_stim[idx], abs(F_stim[idx]), '-k')

                    for k in ['spectrum_base', 'spectrum_stim']:
                        ax[k].set_xticks(np.arange(-2*eod, 3*eod, eod))
                        ax[k].set_xlim((-f_max, f_max))
                        ax[k].set_xticklabels([])
                    ax['spectrum_stim'].set_xticklabels(np.arange(-2, 3))
                    ax['spectrum_stim'].set_xlabel('frequency/EOD')

