import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from locking.analyses import *

rel = Runs()*SecondOrderSignificantPeaks()*StimulusSpikeJitter()*Cells() \
                    - 'cell_id="2014-06-06-ak"' \
                   & dict(eod_coeff=0, baseline_coeff=0, refined=1,\
                          cell_type='p-unit', am=0, n_harmonics=0)\
                   & 'stimulus_coeff > 0'\
                   & 'frequency > 0'\

df = pd.DataFrame(rel.fetch())
print("n={0} cells".format(len(Cells() & rel)))
print("n={0} trials".format(len(Runs() & rel)))
df['spread'] = df['stim_std']/df['eod']/2/np.pi
df['jitter'] = df['stim_std'] # rename to avoid conflict with std function


sns.set_context('paper')

with sns.axes_style('ticks'):
    fig,ax = plt.subplots(1,3, figsize=(7,3), dpi=400, sharey=True)

sc = ax[0].scatter( df.frequency, df.vector_strength, c=df.jitter, cmap=plt.get_cmap('viridis'), edgecolors='w',lw=.5)
fig.colorbar(sc, ax=ax[0])
sc = ax[1].scatter( df.jitter, df.vector_strength,c=df.frequency, cmap=plt.get_cmap('viridis'), edgecolors='w',lw=.5)
fig.colorbar(sc, ax=ax[1])
ax[1].axis('tight')
ax[1].set_xticks(ax[1].get_xticks()[::2])

# mu = df.groupby('contrast').mean().reset_index()
# std = df.groupby('contrast').std().reset_index()
#
# xpos = np.arange(len(mu))
# ax[2].bar( xpos, mu.vector_strength, yerr=std.vector_strength, color='gray', lw=0,
#            ecolor='k', align='center')
# # sns.violinplot('contrast', 'vector_strength', data=df, order=[1.25,2.5,5,10,20], color='gray', ax=ax[2])
#
# ax[2].set_xticks(xpos)
# ax[2].set_xticklabels(mu.contrast)
sns.pointplot('contrast', 'vector_strength', data=df[df.stimulus_coeff==1], order=[2.5,5,10,20], color='gray', ax=ax[2], hue='cell_id',\
               palette='Blues_d', join=True, scale=.5)
ax[2].legend(title=None, ncol=1, fontsize=6, loc='upper left')
ax[2].set_ylabel('')


ax[1].tick_params('y', length=0, width=0)
ax[2].tick_params('y', length=0, width=0)
ax[0].tick_params('y', length=3, width=1)
for a in ax:
    a.tick_params('x', length=3, width=1)

ax[0].set_ylim((0,1))
ax[1].set_xlim((0,3.5))
ax[1].set_xticks([0, 1,2,3])
# sns.despine(ax=ax[1], trim=True, left=True)
ax[2].set_xlim((-0.5, 4.5))
ax[0].set_xlim((0,2400))

ax[0].set_xlabel('frequency [Hz]')
ax[0].set_ylabel('vector strength')
ax[1].set_xlabel('circular std')
ax[2].set_xlabel('contrast [%]')

ax[0].text(-0.3, 1, 'A', transform=ax[0].transAxes, fontweight='bold')
ax[1].text(-0.1, 1, 'B', transform=ax[1].transAxes, fontweight='bold')
ax[2].text(-0.1, 1, 'C', transform=ax[2].transAxes, fontweight='bold')

for a in ax:
    a.tick_params('x', length=3, width=1)
ax[0].tick_params('y', length=3, width=1)


sns.despine(fig, trim=True)
for a in ax:
    for axis in ['top','bottom','left','right']:
        a.spines[axis].set_linewidth(1)

sns.despine(ax=ax[1], left=True)
sns.despine(ax=ax[2], left=True)

fig.tight_layout()
fig.savefig('figures/figure03.pdf')


#glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df, family=sm.families.Gamma()).fit()
glm = smf.glm('vector_strength ~ frequency * jitter + contrast', data=df, family=sm.families.Gamma()).fit()
print(glm.summary())
print(glm.pvalues)

print(r"jitter: \rho={0}".format( np.corrcoef(df.jitter, df.vector_strength)[0,1]))
print(r"frequency: \rho={0}".format( np.corrcoef(df.frequency, df.vector_strength)[0,1]))
print(r"contrast: \rho={0}".format( np.corrcoef(df.contrast, df.vector_strength)[0,1]))

print('1 sigma in Frequency domain', np.mean(1/(2*np.pi*df.spread)))
print('min sigma in Frequency domain', np.min(1/(2*np.pi*df.spread)))
print('max sigma in Frequency domain', np.max(1/(2*np.pi*df.spread)))
print('2 sigma in Frequency domain', np.mean(2/(2*np.pi*df.spread)))
