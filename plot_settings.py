import matplotlib
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rc('font', size=7)
matplotlib.rc('text', usetex='false')
matplotlib.rc('figure', dpi='300')
matplotlib.rc('figure', facecolor='w')



params = {'axes.labelsize': 7,
          'text.fontsize': 7,
          'xtick.labelsize': 7,
          'ytick.labelsize': 7,
          'legend.fontsize': 7,
          'figure.dpi': 300,
          'xtick.major.width': 1,
          'xtick.minor.width': 1,
          'ytick.major.width': 1,
          'ytick.minor.width': 1,
          'xtick.major.size': 3,
          'xtick.minor.size': 3,
          'ytick.major.size': 3,
          'ytick.minor.size': 3,
          'axes.linewidth': 1
}

matplotlib.rcParams.update(params)
