import os
from collections import OrderedDict
import seaborn as sns

colors = [sns.xkcd_rgb['bright pink'], "steelblue", "#e74c3c", sns.xkcd_rgb['apple green'], "gray"]  # "#9a0eea"
colordict = OrderedDict(zip(['stimulus', 'eod', 'baseline', 'delta_f', 'combinations'], colors))
markers = [(4, 0, 90), '^', 'D', 's', 'o']
markerdict = OrderedDict(zip(['stimulus', 'eod', 'baseline', 'delta_f', 'combinations'], markers))

def mkdir(newdir):
    if os.path.isdir(newdir):
        pass
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        if tail:
            os.mkdir(newdir)