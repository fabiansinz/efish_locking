from plot_settings import params as plot_params
import matplotlib.pyplot as plt

from schemata import ISIHistograms, FICurves
# mpl.use('Agg')      # With this line = figure disappears; without this line = warning

from analyses import *
from figure_classes import Figure01

def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)

if __name__ == "__main__":
    f_max = 2000 # Hz
    fos = FirstOrderSpikeSpectra()
    sos = SecondOrderSpikeSpectra()

    runs = Runs()
    for cell in (Cells() & 'cell_id="2014-12-03-aj"').fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        for spectrum, base_name in zip([fos, sos], ['firstorderspectra', 'secondorderspectra']):
            for contrast in [20]:
                print("contrast: %.2f%%" % (contrast,))

                target_trials = ((spectrum & cell & ('contrast = %i' % (contrast,)) & 'am = 0' & 'n_harmonics = 0') * runs)

                if len(target_trials) > 0:
                    with Figure01(filename=generate_filename(cell, contrast=contrast, base=base_name)) as (fig, ax):
                        # --- plot ISI histogram
                        ISIHistograms().plot(ax=ax['ISI'], restrictions=cell)

                        # --- plot FICurves
                        FICurves().plot(ax=ax['FI'], restrictions=cell)

                        # --- plot locking
                        PhaseLockingHistogram().violin_plot(ax['violin'], restrictions=target_trials)

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
                        line_colors = sns.xkcd_rgb['golden yellow'], sns.xkcd_rgb['baby blue'], sns.xkcd_rgb['apple green']

                        ax['spectrum'].plot(stim_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[0],label='stimulus')
                        ax['spectrum'].plot(np.abs(deltaf_freq), y[:-1], '-',  zorder=-1, lw=1, color=line_colors[1], label=r'$|\Delta f|$')
                        ax['spectrum'].plot(eod_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[2], label='EOD')
