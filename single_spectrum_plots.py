import plot_settings
# import matplotlib as mpl
# mpl.use('Agg')  # With this line = figure disappears; without this line = warning
from analyses import *

if __name__ == "__main__":
    f_max = 2000  # Hz
    mkdir('figures/detailedsecondorderspectra')
    SecondOrderSpikeSpectra().plot(figbase='figures/detailedsecondorderspectra', refined=1, am=0, n_harmonics=0)

    mkdir('figures/detailedfirstorderspectra')
    FirstOrderSpikeSpectra().plot(figbase='figures/detailedfirstorderspectra', refined=1, am=0, n_harmonics=0)

    mkdir('figures/phaselocking')
    PhaseLockingHistogram().plot(figbase='figures/phaselocking', refined=1, am=0, n=2, n_harmonics=0)