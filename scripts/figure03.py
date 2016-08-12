import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from locking.analyses import *

# def count(gr):
#     gr['elements'] = len(gr)
#     return gr

rel = Runs() * SecondOrderSignificantPeaks() * StimulusSpikeJitter() * Cells() \
      & dict(eod_coeff=0, baseline_coeff=0, refined=1, \
             cell_type='p-unit', am=0, n_harmonics=0) \
      & 'stimulus_coeff > 0' \
      & 'frequency > 0'

df = pd.DataFrame(rel.fetch())

print("n={0} cells".format(len(Cells() & rel)))
print("n={0} trials".format(len(Runs() & rel)))
df['spread'] = df['stim_std'] / df['eod'] / 2 / np.pi
df['jitter'] = df['stim_std']  # rename to avoid conflict with std function

sns.set_context('paper')

with sns.axes_style('ticks'):
    fig, ax = plt.subplots(1, 3, figsize=(7, 3), dpi=400, sharey=True)
ax = dict(zip(['contrast', 'stimulus', 'cstd'], ax))

sc = ax['stimulus'].scatter(df.frequency, df.vector_strength, c=df.jitter, cmap=plt.get_cmap('viridis'), edgecolors='w',
                            lw=.5)
fig.colorbar(sc, ax=ax['stimulus'])
sc = ax['cstd'].scatter(df.jitter, df.vector_strength, c=df.frequency, cmap=plt.get_cmap('viridis'), edgecolors='w',
                        lw=.5)
fig.colorbar(sc, ax=ax['cstd'])
ax['cstd'].axis('tight')
ax['cstd'].set_xticks(ax['cstd'].get_xticks()[::2])

# df2 = df[df.stimulus_coeff == 1].groupby(['cell_id']).apply(count)

sns.pointplot('contrast', 'vector_strength', data=df[df.stimulus_coeff == 1],
              order=[2.5, 5, 10, 20], color='gray',
              ax=ax['contrast'], hue='cell_id',
              palette='Blues_d', join=True, scale=.5)
ax['contrast'].legend(title=None, ncol=1, fontsize=6, loc='upper left')
ax['contrast'].set_ylabel('vector strength')

ax['cstd'].tick_params('y', length=0, width=0)
ax['contrast'].tick_params('y', length=3, width=1)
ax['stimulus'].tick_params('y', length=0, width=0)
for a in ax.values():
    a.tick_params('x', length=3, width=1)

ax['stimulus'].set_ylim((0, 1))
ax['cstd'].set_xlim((0, 3.5))
ax['cstd'].set_xticks([0, 1, 2, 3])
ax['contrast'].set_xlim((-0.5, 4.5))
ax['stimulus'].set_xlim((0, 2400))

ax['stimulus'].set_xlabel('frequency [Hz]')
ax['cstd'].set_xlabel('circular std')
ax['contrast'].set_xlabel('contrast [%]')

ax['stimulus'].text(-0.1, 1, 'B', transform=ax['stimulus'].transAxes, fontweight='bold')
ax['cstd'].text(-0.1, 1, 'C', transform=ax['cstd'].transAxes, fontweight='bold')
ax['contrast'].text(-0.3, 1, 'A', transform=ax['contrast'].transAxes, fontweight='bold')

for a in ax.values():
    a.tick_params('x', length=3, width=1)

sns.despine(fig, trim=True)
for a in ax.values():
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(1)

sns.despine(ax=ax['cstd'], left=True)
sns.despine(ax=ax['stimulus'], left=True)

fig.tight_layout()
fig.savefig('figures/figure03.pdf')

# glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df, family=sm.families.Gamma()).fit()
glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df, family=sm.families.Gamma()).fit()
print(glm.summary())
print(glm.pvalues)

print(r"jitter: \rho={0}".format(np.corrcoef(df.jitter, df.vector_strength)[0, 1]))
print(r"frequency: \rho={0}".format(np.corrcoef(df.frequency, df.vector_strength)[0, 1]))
print(r"contrast: \rho={0}".format(np.corrcoef(df.contrast, df.vector_strength)[0, 1]))

print('1 sigma in Frequency domain', np.mean(1 / (2 * np.pi * df.spread)))
print('min sigma in Frequency domain', np.min(1 / (2 * np.pi * df.spread)))
print('max sigma in Frequency domain', np.max(1 / (2 * np.pi * df.spread)))
print('2 sigma in Frequency domain', np.mean(2 / (2 * np.pi * df.spread)))
