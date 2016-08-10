from locking import analyses as ana
from locking import data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datajoint as dj

# freqs = data.Cells() * data.Runs() * ana.SecondOrderSignificantPeaks() & dict(cell_type='p-unit', am=0,
#                                                                               refined=True) & ['contrast=20',
#                                                                                                'contrast=10']
#
# stim_freqs = freqs & dict(stimulus_coeff=1, eod_coeff=0, baseline_coeff=0)
# df_freqs = freqs & dict(stimulus_coeff=1, eod_coeff=-1, baseline_coeff=0)
#
# dat = (dj.U('delta_f', 'contrast') * stim_freqs.proj('delta_f', 'contrast', vs_stim='vector_strength', e1='eod_coeff',
#                                                      s1='stimulus_coeff')) * \
#       df_freqs.proj('delta_f', 'contrast', vs_df='vector_strength', e2='eod_coeff', s2='stimulus_coeff')
#
# # df = pd.DataFrame(dat.fetch())#.groupby(['delta_f','contrast']).mean().reset_index()
# df_stim = pd.DataFrame(stim_freqs.proj('delta_f', 'contrast', 'vector_strength').fetch())
# df_stim['stimulus'] = 1
# df_df = pd.DataFrame(df_freqs.proj('delta_f', 'contrast', 'vector_strength').fetch())
# df_df['stimulus'] = 0
#
# df = pd.concat([df_stim, df_df])

# freqs = (ana.Decoding()*ana.Decoding.Beat()*ana.Decoding.Stimulus()*ana.Runs()) & 'contrast between 10 and 20'
# stim_freqs = freqs & dict(stimulus_coeff=1, eod_coeff=0, baseline_coeff=0)
# df_freqs = freqs & dict(stimulus_coeff=1, eod_coeff=-1, baseline_coeff=0)
# df_stim = pd.DataFrame(stim_freqs.proj('contrast', vector_strength='vs_stimulus', delta_f='beat').fetch())
# df_stim['locking frequency'] = 'stimulus'
# df_df = pd.DataFrame(df_freqs.proj('contrast', vector_strength='vs_beat', delta_f='beat').fetch())
# df_df['locking frequency'] = 'beat'
# df = pd.concat([df_stim, df_df])
#
# # bins = [-501, -400, -300, -200, -100, 0, 100, 200,300, 400, 501]
# bins = [-501, -250, 0, 250, 501]
# df_group = np.array(['(%.0f, %.0f)' % k for k in zip(bins[:-1], bins[1:])], dtype=object)
#
# df['delta_f bin'] = df_group[np.digitize(df.delta_f, bins) - 1]
#
# sns.factorplot('delta_f bin', 'vector_strength', hue='locking frequency', kind='bar', data=df, order=df_group, col='contrast')
# plt.show()


fig, ax = plt.subplots(1,2)

for a, contrast in zip(ax, [10,20]):
    vs, df =  (ana.Decoding()*ana.Decoding.Stimulus()*ana.Runs() & dict(contrast=contrast)).fetch['vs_stimulus','beat']
    a.plot(df, vs, 'o', label='to stimulus', mfc='gold')
    vs, df =  (ana.Decoding()*ana.Decoding.Beat()*ana.Runs() & dict(contrast=contrast)).fetch['vs_beat','beat']
    a.plot(df, vs,'o', label='to beat', mfc='steelblue')
    a.legend()
plt.show()
