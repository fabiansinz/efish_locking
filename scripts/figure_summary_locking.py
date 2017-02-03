from matplotlib_venn import venn3

from locking import mkdir
from locking import data
from locking import analyses as ana
from locking.data import ISIHistograms, FICurves
from scripts.config import params as plot_params, FormatedFigure
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')
from locking.analyses import *




# class FigureSummary(FormatedFigure):
#     def prepare(self):
#         sns.set_style('ticks')
#         sns.set_context('paper')
#         with plt.rc_context(plot_params):
#             self.fig = plt.figure(figsize=(7, 7))
#             gs = plt.GridSpec(3, 2)
#             self.ax = {
#                 'ispectrum': self.fig.add_subplot(gs[2, :]),
#                 'scatter': self.fig.add_subplot(gs[1, :]),
#                 # 'spectrum': self.fig.add_subplot(gs[1:, :-1]),
#                 'ISI': self.fig.add_subplot(gs[0, 0]),
#                 'EOD': self.fig.add_subplot(gs[0, 1]),
#                 # 'FI': self.fig.add_subplot(gs[0, 2]),
#             }
#             # self.ax['violin'] = self.fig.add_subplot(gs[1:, -1])
#         self.gs = gs
#
#     @staticmethod
#     def format_ISI(ax):
#         sns.despine(ax=ax, left=True)
#         ax.set_yticks([])
#         ax.text(-0.15, 1.01, 'A', transform=ax.transAxes, fontweight='bold')
#
#     # @staticmethod
#     # def format_FI(ax):
#     #     sns.despine(ax=ax)
#     #     ax.text(-0.1, 1.01, 'C', transform=ax.transAxes, fontweight='bold')
#     #     ax.legend(loc='upper right')
#
#     @staticmethod
#     def format_EOD(ax):
#         sns.despine(ax=ax, left=True, trim=True)
#         ax.text(-0.1, 1.01, 'B', transform=ax.transAxes, fontweight='bold')
#
#     @staticmethod
#     def format_ispectrum(ax):
#         sns.despine(ax=ax, trim=True)
#         ax.text(-0.06, 1.01, 'D', transform=ax.transAxes, fontweight='bold')
#
#     @staticmethod
#     def format_scatter(ax):
#         sns.despine(ax=ax)
#         ax.set_xlabel('time [EOD cycles]')
#         ax.text(-0.06, 1.02, 'C', transform=ax.transAxes, fontweight='bold')
#
#     def format_figure(self):
#         self.fig.tight_layout()
#         self.gs.tight_layout(self.fig)


if __name__ == "__main__":
    restr = 'abs(stimulus_coeff) + abs(eod_coeff) + abs(baseline_coeff) = 1'
    # (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    df = -400
    cells = data.Cells() #& 'cell_type="p-unit"'
    peaks = data.Runs() * ana.SecondOrderSignificantPeaks() & 'n_harmonics=0 and refined=1 and am=0' \
                & dict(delta_f=df)
    eod = set(cells.aggregate(peaks \
                              & 'stimulus_coeff=0 and eod_coeff=1 and baseline_coeff=0',
                              count='count(eod)').fetch['cell_id'])
    stimulus = set(cells.aggregate(peaks \
                              & 'stimulus_coeff=1 and eod_coeff=0 and baseline_coeff=0',
                              count='count(eod)').fetch['cell_id'])
    delta = set(cells.aggregate(peaks \
                              & 'stimulus_coeff=1 and eod_coeff=-1 and baseline_coeff=0',
                              count='count(eod)').fetch['cell_id'])
    print(delta, eod, stimulus)
    venn3([eod, stimulus, delta], set_labels = ('EODf', 'stimulus', r'$\Delta f$'))
    plt.gcf().savefig('test.png')