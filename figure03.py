from schemata import *
from analyses import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np

df = pd.DataFrame((Runs()*SecondOrderSignificantPeaks()*SpikeJitter()*Cells()
                   & dict(eod_coeff=0, baseline_coeff=0, refined=1,
                          cell_type='p-unit', am=0, n_harmonics=0)
                   & 'stimulus_coeff > 0'
                   & 'frequency > 0').fetch())
df['spread'] = np.sqrt(df['var'])/2/np.pi/df['eod']*1000
sns.set_context('paper')

with sns.axes_style('ticks'):
    fig,ax = plt.subplots(1,3, figsize=(7,3), dpi=400, sharey=True)

ax[0].scatter( df.frequency, df.vector_strength, color='gray')
ax[1].scatter( df.spread, df.vector_strength, color='gray')
ax[1].axis('tight')
ax[1].set_xticks(ax[1].get_xticks()[::2])

mu = df.groupby('contrast').mean().reset_index()
std = df.groupby('contrast').std().reset_index()

xpos = np.arange(len(mu))
ax[2].bar( xpos, mu.vector_strength, yerr=std.vector_strength, color='gray', lw=0,
           ecolor='k', align='center')

ax[2].set_xticks(xpos)
ax[2].set_xticklabels(mu.contrast)


sns.despine(fig)
sns.despine(ax=ax[1], left=True)
sns.despine(ax=ax[2], left=True)

ax[1].tick_params('y', length=0, width=0)
ax[2].tick_params('y', length=0, width=0)
ax[0].tick_params('y', length=3, width=1)
for a in ax:
    a.tick_params('x', length=3, width=1)
ax[0].set_ylim((0,1))
ax[2].set_xlim((-0.5, 4.5))
ax[0].set_xlim((0,2100))

ax[0].set_xlabel('frequency [Hz]')
ax[0].set_ylabel('vector strength')
ax[1].set_xlabel('jitter [ms]')
ax[2].set_xlabel('contrast [%]')

ax[0].text(-0.3, 1, 'A', transform=ax[0].transAxes, fontweight='bold')
ax[1].text(-0.1, 1, 'B', transform=ax[1].transAxes, fontweight='bold')
ax[2].text(-0.1, 1, 'C', transform=ax[2].transAxes, fontweight='bold')

fig.tight_layout()
fig.savefig('figures/figure03.pdf')


glm = smf.glm('vector_strength ~ frequency + spread + contrast', data=df, family=sm.families.Gamma()).fit()
print(glm.summary())
print(np.corrcoef(df.spread, df.vector_strength))

print('1 sigma in Frequency domain', 1/(2*np.pi*np.mean(df.spread/1000)))
print('2 sigma in Frequency domain', 2/(2*np.pi*np.mean(df.spread/1000)))