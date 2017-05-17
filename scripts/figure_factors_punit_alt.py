import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from locking.analyses import *

rel = Runs() * SecondOrderSignificantPeaks() * StimulusSpikeJitter() * Cells() \
      & dict(stimulus_coeff=1, eod_coeff=0, baseline_coeff=0, refined=1, \
             cell_type='p-unit', am=0, n_harmonics=0) \
      & 'frequency > 0'
# & 'stimulus_coeff = 1' \



df = pd.DataFrame(rel.fetch())

print("n={0} cells".format(len(Cells() & rel)))
print("n={0} trials".format(len(Runs() & rel)))
df['spread'] = df['stim_std'] / df['eod'] / 2 / np.pi
df['jitter'] = df['stim_std']  # rename to avoid conflict with std function

sns.set_context('paper')

with sns.axes_style('ticks'):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=400, sharey=True)
ax = dict(zip(['contrast', 'stimulus'], ax))

# =============================================================================
# -- plot contrast vs. vector strength

sns.pointplot('contrast', 'vector_strength', data=df[df.stimulus_coeff == 1],
              order=[2.5, 5, 10, 20],
              ax=ax['contrast'], hue='cell_id',
              palette={ci: sns.xkcd_rgb['azure'] for ci in pd.unique(df.cell_id)},
              join=True, scale=.5)
leg = ax['contrast'].legend(title=None, ncol=1, fontsize=6, loc='upper left')
leg.set_visible(False)
ax['contrast'].set_ylabel('vector strength at stimulus')
ax['contrast'].tick_params('y', length=3, width=1)
ax['contrast'].set_xlim((-.2, 3.5))
ax['contrast'].set_xlabel('contrast [%]')
ax['contrast'].text(-0.17, 1.05, 'A', transform=ax['contrast'].transAxes, fontweight='bold')

# =============================================================================
# --- statistical analysis

glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df, family=sm.families.Gamma()).fit()

print(glm.summary())
print(glm.pvalues)

print('1 sigma in Frequency domain', np.mean(1 / (2 * np.pi * df.spread)))
print('min sigma in Frequency domain', np.min(1 / (2 * np.pi * df.spread)))
print('max sigma in Frequency domain', np.max(1 / (2 * np.pi * df.spread)))
print('2 sigma in Frequency domain', np.mean(2 / (2 * np.pi * df.spread)))

# =============================================================================
print(r"contrast: \rho={0}    p={1}".format(*stats.pearsonr(df.contrast, df.vector_strength)))
df = df[df.contrast == 20]
print(r"jitter: \rho={0}    p={1}".format(*stats.pearsonr(df.jitter, df.vector_strength)))
print(r"frequency: \rho={0}    p={1}".format(*stats.pearsonr(df.frequency, df.vector_strength)))

# =============================================================================
# --- plot frequency vs. vector strength


bins = [0, np.pi / 5, 2 * np.pi / 5, np.inf]
labels = [r'0 - $\frac{\pi}{5}$', r'$\frac{\pi}{5}$ - $2\cdot\frac{\pi}{5}$', r'$2\cdot\frac{\pi}{5}$ - $\infty$']
color =sns.color_palette("BrBG", 3)
color[1] = 'slategray'
# for low, high, col, lab in zip(bins[:-1], bins[1:], color, labels):
#     idx = (df.jitter >= low) &  (df.jitter < high)
#     sc = ax['stimulus'].scatter(df.frequency[idx], df.vector_strength[idx], color=col, edgecolors='w', lw=.5, s=30,
#                                 label=lab)
# color = sns.color_palette('BrBG', len(pd.unique(df.cell_id)))
color = sns.blend_palette(['teal', 'steelblue','gold','slategray','brown'], len(pd.unique(df.cell_id)))
for (cell_id, cell), col in zip(df.groupby('cell_id'), color):
    # if cell_id != '2014-11-13-aa':
    sc = ax['stimulus'].scatter(cell.jitter, cell.vector_strength, color=col, edgecolors='w', lw=.5, s=20,
                                    label=cell_id)


# ax['stimulus'].legend(title='circular std',
#                     fontsize=ax['stimulus'].xaxis.get_ticklabels()[0].get_fontsize(),
#                      ncol=1, bbox_to_anchor=(1.05,1.15))
ax['stimulus'].legend(title='circular std',
                    fontsize=ax['stimulus'].xaxis.get_ticklabels()[0].get_fontsize(),
                     ncol=1, bbox_to_anchor=(1.35,1.15))
ax['stimulus'].set_ylim((0, 1))

ax['stimulus'].tick_params('y', length=0, width=0)

# ax['stimulus'].set_xlim((0, 1800))
# ax['stimulus'].set_xticks(np.arange(0, 2000, 500))

ax['stimulus'].set_xlabel('frequency [Hz]')
ax['stimulus'].text(-0.1, 1.05, 'B', transform=ax['stimulus'].transAxes, fontweight='bold')

# =============================================================================
for a in ax.values():
    a.tick_params('x', length=3, width=1)

sns.despine(fig, trim=True)
for a in ax.values():
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(1)

sns.despine(ax=ax['stimulus'], left=True)

fig.tight_layout()
fig.subplots_adjust(top=0.9, right=0.85)
fig.savefig('figures/figure_factors_punit_alt.pdf')
fig.savefig('figures/figure_factors_punit_alt.png')
