"""Plotting functions and classes"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants

from matplotlibstyles import styles


def setup_figure():
    styles.set_thin_latex_style()
    figsize = (styles.cm_to_inches(8.6), styles.cm_to_inches(6))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)

def save_figure(f, plot_filebase):
    f.savefig(plot_filebase + '.pgf', transparent=True)
    f.savefig(plot_filebase + '.pdf', transparent=True)
    f.savefig(plot_filebase + '.png', transparent=True)


class Plot:
    def __init__(self, args):
        self._args = args

    def set_labels(self, ax):
        plt.legend()


class FreqsPlot(Plot):
    def plot_figure(self, f, ax):
        itr = self._args['itr']
        input_dir = self._args["input_dir"]
        for vari in self._args['varis']:
            for rep in range(1, self._args['reps'] + 1):
                filename = f'{input_dir}/{vari}/{vari}_rep-{rep}.freqs'
                freqs = pd.read_csv(filename, header=0, delim_whitespace=True)
                heights = freqs.columns.astype(int)

                #ax.plot(heights, np.log10(freqs.iloc[itr - 1]))
                ax.plot(heights, freqs.iloc[itr - 1])

    def setup_axis(self, ax):
        ax.set_ylabel(r'Fraction')
        ax.set_xlabel(r'Lattice height')


class LFEsPlot(Plot):
    def plot_figure(self, f, ax):
        itr = self._args['itr']
        input_dir = self._args["input_dir"]
        for vari in self._args['varis']:
            for rep in range(1, self._args['reps'] + 1):
                filename = f'{input_dir}/{vari}/{vari}_rep-{rep}.biases'
                biases = pd.read_csv(filename, header=0, delim_whitespace=True)
                heights = biases.columns.astype(int)
                lfes = -biases / (self._args['temp']*constants.k)

                ax.plot(heights, lfes.iloc[itr - 1], marker='o')

    def setup_axis(self, ax):
        ax.set_ylabel(r'$k_\mathrm{b}T$')
        ax.set_xlabel('Lattice height')


class RadiiPlot(Plot):
    def plot_figure(self, f, ax):
        vari = self._args['vari']
        input_dir = self._args["input_dir"]
        rep = self._args['rep']
        itr = self._args['itr']
        filename = f'{input_dir}/{vari}/{vari}_rep-{rep}_iter-{itr}.ops'
        ops = pd.read_csv(filename, header=0, delim_whitespace=True)
        ax.plot(ops['step'], ops['radius'])

    def setup_axis(self, ax):
        ax.set_ylabel(r'Radius')
        ax.set_xlabel(r'Step')
