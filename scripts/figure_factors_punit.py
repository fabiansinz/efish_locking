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
    fig, ax = plt.subplots(1, 3, figsize=(7, 3), dpi=400, sharey=True)
ax = dict(zip(['contrast', 'stimulus', 'cstd'], ax))

# =============================================================================
# -- plot contrast vs. vector strength

sns.pointplot('contrast', 'vector_strength', data=df[df.stimulus_coeff == 1],
              # order=[2.5, 5, 10, 20],
              order=[10, 20],
              ax=ax['contrast'], hue='cell_id',
              palette={ci: sns.xkcd_rgb['azure'] for ci in pd.unique(df.cell_id)},
              join=True, scale=.5)
leg = ax['contrast'].legend(title=None, ncol=1, fontsize=6, loc='upper left')
leg.set_visible(False)
ax['contrast'].set_ylabel('vector strength at stimulus')
ax['contrast'].tick_params('y', length=3, width=1)
# ax['contrast'].set_xlim((-.2, 3.5))
ax['contrast'].set_xlabel('contrast [%]')
ax['contrast'].text(-0.3, 1, 'A', transform=ax['contrast'].transAxes, fontweight='bold')

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
sc = ax['stimulus'].scatter(df.frequency, df.vector_strength, c=df.jitter, cmap=plt.get_cmap('viridis'), edgecolors='w',
                            lw=.5)
cb = fig.colorbar(sc, ax=ax['stimulus'])
cb.set_ticks((np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3))
cb.set_ticklabels([r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$'])
cb.set_label('circular std [unit circle circumference]', fontsize=8)
ax['stimulus'].set_ylim((0, 1))
ax['stimulus'].tick_params('y', length=0, width=0)
ax['stimulus'].set_xlim((0, 1800))
ax['stimulus'].set_xticks(np.arange(0, 2000, 500))

ax['stimulus'].set_xlabel('frequency [Hz]')
ax['stimulus'].text(-0.1, 1, 'B', transform=ax['stimulus'].transAxes, fontweight='bold')

# =============================================================================
# --- plot jitter vs. vector strength
sc = ax['cstd'].scatter(df.jitter, df.vector_strength, c=df.frequency, cmap=plt.get_cmap('viridis'), edgecolors='w',
                        lw=.5)
cb = fig.colorbar(sc, ax=ax['cstd'])
cb.set_label('stimulus frequency [Hz]', fontsize=8)
# t = np.linspace(1e-6, 1-1e-6,100)
ax['cstd'].set_xlim((0, 2.8))
ax['cstd'].set_xticks((0, np.pi / 4, np.pi / 2, 3 / 4 * np.pi))
ax['cstd'].set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', ])
ax['cstd'].tick_params('y', length=0, width=0)
ax['cstd'].set_xlabel('circular std at EODf')
# ax['cstd'].plot(np.sqrt(-2*np.log(t)), t , '--k')
ax['cstd'].text(-0.1, 1, 'C', transform=ax['cstd'].transAxes, fontweight='bold')

# =============================================================================
for a in ax.values():
    a.tick_params('x', length=3, width=1)

sns.despine(fig, trim=True)
for a in ax.values():
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(1)

sns.despine(ax=ax['cstd'], left=True)
sns.despine(ax=ax['stimulus'], left=True)

fig.tight_layout()
fig.savefig('figures/figure_factors_punit.pdf')
fig.savefig('figures/figure_factors_punit.png')
