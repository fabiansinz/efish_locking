from IPython import embed
import matplotlib as mpl
from helpers import mkdir

mpl.use('Agg')      # With this line = figure disappears; without this line = warning

import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from schemata import *
from analyses import *
from figure_classes import MultiSpectrumFigure
import os


def generate_filename(cell, contrast, base='firstorderspectra'):
    dir = 'figures/%s/%s/' % (base, cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.pdf' % (cell['cell_id'], contrast)

if __name__ == "__main__":
    f_max = 2000 # Hz
    fos = FirstOrderSpikeSpectra()
    sos = FirstOrderSpikeSpectra()

    runs = Runs()
    for cell in Cells():
        unit = cell['cell_type']
        print('Processing', cell['cell_id'])
        for df, rel in (Runs() & cell & 'am = 0').group_by('delta_f'):
            print(u"\t\u0394 f=%.2f" % df)
            # for s, base in zip([fos, sos], ['firstorderspectra', 'secondorderspectra']):
            #     contrasts = sorted(np.unique((Runs() & cell & 'am = 0' & 'delta_f').project('contrast').fetch()['contrast']))
            #     for contrast in [10, 20]:
            #         target_trials = ((s & cell & ('contrast = %i' % (contrast,)) & 'am = 0' & 'n_harmonics = 0') * runs)
            #         if len(target_trials) > 0:
            #             with SpectrumFigure(filename=generate_filename(cell, contrast=contrast, base=base)) as (fig, ax):
            #                 y = [0]
            #                 stim_freq, eod_freq, deltaf_freq = [], [], []
            #                 for i, spec in enumerate(sorted(target_trials.fetch(as_dict=True), key=lambda x: x['delta_f'])):
            #
            #                     f, v = spec['frequencies'], spec['vector_strengths']
            #                     idx = (f >= 0) & (f <= f_max) & ~np.isnan(v)
            #                     ax.fill_between(f[idx], y[-1] + 0*f[idx], y[-1] + v[idx], lw=0, color='darkslategray')
            #
            #                     y.append(y[-1] + .8)
            #                     stim_freq.append(spec['eod'] + spec['delta_f'])
            #                     deltaf_freq.append(spec['delta_f'])
            #                     eod_freq.append(spec['eod'])
            #                 line_colors = sns.xkcd_rgb['golden yellow'], sns.xkcd_rgb['baby blue'], sns.xkcd_rgb['apple green']
            #
            #                 ax.plot(stim_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[0],label='stimulus')
            #                 ax.plot(np.abs(deltaf_freq), y[:-1], '-',  zorder=-1, lw=1, color=line_colors[1], label=r'$|\Delta f|$')
            #                 ax.plot(eod_freq, y[:-1], '-', zorder=-1, lw=1, color=line_colors[2], label='EOD')

