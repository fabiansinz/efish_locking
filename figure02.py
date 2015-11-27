import os
from datajoint.utils import user_choice
from figure01 import generate_filename
from plot_settings import params as plot_params
import matplotlib.pyplot as plt

from figure_classes import Figure02
from analyses import *


def generate_filename(cell, contrast):
    dir = 'figures/figure02/%s/' % (cell['cell_type'], )
    mkdir(dir)
    return dir + '%s_contrast%.2f.png' % (cell['cell_id'], contrast)

if __name__ == "__main__":
    f_max = 2000 # Hz
    fos = FirstOrderSpikeSpectra()
    sos = SecondOrderSpikeSpectra()

    runs = Runs()
    for cell in (Cells() & 'cell_id="2014-12-03-ao"').fetch.as_dict:

        unit = cell['cell_type']
        print('Processing', cell['cell_id'])

        line_colors = sns.color_palette('pastel', n_colors=3)#sns.xkcd_rgb['golden yellow'], sns.xkcd_rgb['baby blue'], sns.xkcd_rgb['apple green']
        for contrast in [20]:#[5, 10, 20]:
            print("contrast: %.2f%%" % (contrast,))

            target_trials = runs & cell & dict(contrast=contrast, am=0, n_harmonics=0)

            if target_trials:
                with Figure02(filename=generate_filename(cell, contrast=contrast)) as (fig, ax):
                    # --- plot ISI histogram
                    EODStimulusPSTSpikes().plot(ax=ax['psth'], restrictions=target_trials)

