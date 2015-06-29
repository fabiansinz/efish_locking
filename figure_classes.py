__author__ = 'fabee'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from schemata import *
from analyses import *



class MultiSpectrumFigure:

    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        self.fig, self.ax = plt.subplots(facecolor='w')
        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):

        fig, ax = self.fig, self.ax
        ax.set_xlim((0,2400))
        ax.legend(loc='best')
        # ax.set_ylim((0,1))
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        ax.set_yticks([])
        ax.set_xlabel('frequency [Hz]')
        fig.tight_layout()
        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self

class MultiContrastFigure:

    def __init__(self, filename=None):
        self.filename = filename

    def __enter__(self):
        sns.set_style('ticks')
        sns.set_context('paper')
        self.fig, self.ax = plt.subplots(facecolor='w')
        return self.fig, self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):

        fig, ax = self.fig, self.ax
        ax.set_xlim((0,2400))
        ax.legend(loc='best')
        # ax.set_ylim((0,1))
        sns.despine(ax=ax, left=True, trim=True, offset=5)
        ax.set_yticks([])
        ax.set_xlabel('frequency [Hz]')
        fig.tight_layout()
        if self.filename is not None:
            self.fig.savefig(self.filename)

    def __call__(self, *args, **kwargs):
        return self
