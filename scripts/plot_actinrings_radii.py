#!/usr/bin/env python

"""Plot radii at each time step"""

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
    vari = args['vari']
    rep = args['rep']
    itr = args['itr']
    filename = f'{args["input_dir"]}/{vari}/{vari}_rep-{rep}_iter-{itr}.ops'
    ops = pd.read_csv(filename, header=0, delim_whitespace=True)
    ax.plot(ops['step'], ops['radius'])


def setup_axis(ax):
    ax.set_ylabel(r'Radius')
    ax.set_xlabel(r'Step')


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
        'vari',
        type=str,
        help='Variant name')
    parser.add_argument(
        'rep',
        type=int,
        help='Replicate number')
    parser.add_argument(
        'itr',
        type=int,
        help='Iteration number')

    return parser.parse_args()


if __name__ == '__main__':
    main()
