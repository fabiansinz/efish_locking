from locking import mkdir, data, analyses
import datajoint as dj
import seaborn as sns
import matplotlib.pyplot as plt
import os

mkdir('figures/sanity/alignments')


# for run in ((data.Runs()*data.Cells()).project('am','n_harmonics','cell_type') & dict(am=0, n_harmonics=0, cell_type='p-unit')).fetch.as_dict():
#     with sns.axes_style('whitegrid'):
#         fig, ax = plt.subplots()
#     analyses.TrialAlign().plot_traces(ax, run)
#     ax.set_title('{cell_id} {cell_type} {run_id}'.format(**run))
#
#     fig.savefig('figures/sanity/alignments/{cell_id}_{run_id}.png'.format(**run))
#     plt.close(fig)

for cell in (data.Cells() & dict(cell_type='p-unit')).fetch.as_dict():
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots()
    analyses.TrialAlign().plot(ax, cell)
    ax.set_title('{cell_id} {cell_type}'.format(**cell))

    fig.savefig('figures/sanity/alignments/{cell_id}.pdf'.format(**cell))
    plt.close(fig)