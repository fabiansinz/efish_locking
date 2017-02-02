import os
from collections import OrderedDict
import seaborn as sns

colors = ["#ff474c", "steelblue", "#e74c3c", sns.xkcd_rgb['sunflower yellow'], "gray"]  # "#9a0eea"
colordict = OrderedDict(zip(['stimulus', 'eod', 'baseline', 'delta_f', 'combinations'], colors))

def mkdir(newdir):
    if os.path.isdir(newdir):
        pass
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        if tail:
            os.mkdir(newdir)