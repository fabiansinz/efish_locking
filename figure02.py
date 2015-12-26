import os
from datajoint.utils import user_choice
from figure01 import generate_filename
from plot_settings import params as plot_params
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift
from figure_classes import Figure02
from analyses import *
from schemata import *

def generate_filename(cell, contrast):
    dir = 'figures/figure02/%s/' % (cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)

def gauss(t, m, v):
    return np.exp(-(t-m)**2/2/v)

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
                    b = Baseline() & cell
                    Baseline().plot_psth(ax['baseline'], cell)

                    # --- plot time cartoon psth baseline
                    eod = (Baseline() & cell).fetch1['eod']
                    stim_period = 1/(eod-delta_f)
                    var = (1/8/eod)**2
                    t = np.linspace(-N/eod,N/eod, 10000)
                    base = lambda t: np.cos(2*np.pi*eod*t)+1
                    beat = lambda t: np.cos(2*np.pi*delta_f*t)+1
                    stim = lambda t: np.cos(2*np.pi*eod*t)+np.cos(2*np.pi*(eod-delta_f)*t) + 2
                    f_base = sum(gauss(t, mu, var) for mu in np.arange(-N/eod,N/eod,1/eod))
                    f_stim = sum(gauss(t, mu, var)*beat(mu) for mu in np.arange(-N/eod,N/eod,1/eod))

                    ax['period'].fill_between(t/stim_period*2*np.pi, f_stim*0, f_stim,color='deeppink')
                    ax['period'].fill_between(t/stim_period*2*np.pi, f_base*0, f_base,color='dodgerblue')
                    ax['period'].set_yticks([])

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
