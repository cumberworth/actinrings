#!/usr/bin/env python

"""Plot LFEs along radius from US weights"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from scipy import constants

from matplotlibstyles import styles


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0, 0])

    plot_figure(f, ax, vars(args))
    setup_axis(ax)
    # set_labels(ax)
    save_figure(f, args.plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (styles.cm_to_inches(10), styles.cm_to_inches(7))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, args):
    for vari in args['varis']:
        filename = f'{args["input_dir"]}/{vari}/{vari}.biases'
        biases = pd.read_csv(filename, header=0, delim_whitespace=True)
        heights = biases.columns.astype(int)
        lfes = -biases / (args['temp']*constants.k)
        ax.plot(heights, lfes.iloc[0])
        ax.plot(heights, lfes.iloc[-1])


def setup_axis(ax):
    ax.set_ylabel(r'$k_\mathrm{b}T$')
    ax.set_xlabel('Lattice height')


def set_labels(ax):
    plt.legend()


def save_figure(f, plot_filebase):
    #f.savefig(plot_filebase + '.pgf', transparent=True)
    f.savefig(plot_filebase + '.pdf', transparent=True)
    f.savefig(plot_filebase + '.png', transparent=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory')
    parser.add_argument(
        'plot_filebase',
        type=str,
        help='Plots filebase')
    parser.add_argument(
        'temp',
        type=float,
        help='Simulation temperature')
    parser.add_argument(
        '--varis',
        nargs='+',
        type=str,
        help='Simulation variants')

    return parser.parse_args()


if __name__ == '__main__':
    main()
