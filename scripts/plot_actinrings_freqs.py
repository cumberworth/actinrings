#!/usr/bin/env python

"""Plot frequencies along radius"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from actinrings import plots


def main():
    args = vars(parse_args())
    p = plots.FreqsPlot(args)
    f = plots.setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0, 0])

    p.plot_figure(f, ax)
    p.setup_axis(ax)
    # p.set_labels(ax)
    plots.save_figure(f, args['plot_filebase'])


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
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'itr',
        type=int,
        help='Iteration number')
    parser.add_argument(
        '--varis',
        nargs='+',
        type=str,
        help='Simulation variants')

    return parser.parse_args()


if __name__ == '__main__':
    main()