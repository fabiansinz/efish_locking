import abc

params = {'axes.labelsize': 7,
          'axes.labelpad': 1.0,
          'axes.titlesize': 7,
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
          'xtick.major.pad': 3,
          'xtick.minor.pad': 3,
          'ytick.major.pad': 3,
          'ytick.minor.pad': 3,
          'xtick.minor.size': 2,
          'ytick.major.size': 2,
          'ytick.minor.size': 3,
          'axes.linewidth': 1,
          'font.family': 'sans-serif',
          'font.serif': 'Arial',
          'font.size': 7,
          'text.usetex': False,
          'figure.dpi': 300,
          'figure.facecolor': 'w',
          }


class FormatedFigure:
    def __init__(self, filename=None):
        self.filename = filename

    @abc.abstractmethod
    def prepare(self):
        pass

    def __enter__(self):
        self.prepare()
        return self.fig, self.ax

    def format_figure(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, ax in self.ax.items():
            if hasattr(self, 'format_' + k):
                getattr(self, 'format_' + k)(ax)

        self.format_figure()
        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self
