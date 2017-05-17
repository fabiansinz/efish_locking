from locking import analyses as ana
import datajoint as dj
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

lock = dj.U('cell_id', 'contrast', 'delta_f') & (ana.Cells() * ana.Runs() * ana.SecondOrderSignificantPeaks() \
                                                 & dict(cell_type='p-unit', stimulus_coeff=1, baseline_coeff=0,
                                                        eod_coeff=0, am=0, n_harmonics=0) \
                                                 & 'MOD(delta_f, 100) = 0 and contrast >= 10')

tested = dj.U('cell_id', 'contrast', 'delta_f') & (ana.Cells() * ana.Runs() \
                                                   & dict(cell_type='p-unit', stimulus_coeff=1, baseline_coeff=0,
                                                          eod_coeff=0, am=0, n_harmonics=0) \
                                                   & 'MOD(delta_f, 100) = 0 and contrast >= 10')

df_num = pd.DataFrame(lock.fetch())
df_denom = pd.DataFrame(tested.fetch())
# df_num['locking'] = 1

gr = ['contrast', 'delta_f']
df_num = df_num.groupby(gr).count()
df_denom = df_denom.groupby(gr).count()
perc = df_num / df_denom * 100
perc = perc.reset_index()
perc.columns = ['contrast', 'delta_f', 'lock']
perc['delta_f'] = ['%.0f' % c for c in perc['delta_f']]
perc['contrast'] = ['%.0f%%' % c for c in perc['contrast']]
sns.set_context('paper')

dfn = np.arange(-500, 600,100)
df = ['%.0f' % c for c in dfn]
with sns.axes_style('ticks'):
    fig, ax = plt.subplots(2, 1, sharex=True)
perc = perc.set_index(['contrast','delta_f'])

df_num = df_num.reset_index()
df_num['contrast'] = ['%.0f%%' % c for c in df_num['contrast']]
df_num = df_num.set_index(['contrast','delta_f'])
w = 30
for contrast, shift, color in zip(['10%', '20%'], [-1,+1], ['lightgrey','grey']):

    ax[0].bar(dfn - shift*w/2, df_num.ix[contrast].ix[dfn,'cell_id'], align='center', width=w, lw=0, color=color,
              label=contrast)
    ax[1].bar(dfn - shift*w/2, perc.ix[contrast].ix[df,'lock'], align='center', width=w, lw=0, color=color)

ax[0].legend(title='contrast')
ax[1].set_xticks(dfn)
ax[1].set_xticklabels(df)
ax[1].set_xlabel(r'$\Delta f$ [Hz]')
ax[0].set_ylabel('locking cells')
ax[1].set_ylabel('locking cells [% tested]')
sns.despine(fig, trim=True)
for a in ax:
    a.tick_params('both', length=3, width=1, which='both')
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(1)


fig.subplots_adjust(bottom=.2)
plt.savefig('figures/summary.pdf')
