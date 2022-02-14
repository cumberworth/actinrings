#!/usr/bin/env python

"""Plot LFEs along radius from US weights.

Plot a curve for each replicate of every variant specfied. It assumes that the given
variant output files are stored in a directory with the same names as the variant.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from actinrings import plots


def main():
    args = vars(parse_args())
    p = plots.LFEsPlot(args)
    f = plots.setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0, 0])

    p.plot_figure(ax)
    p.setup_axis(ax)
    # p.set_labels(ax)
    plots.save_figure(f, args["plot_filebase"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("plot_filebase", type=str, help="Plots filebase")
    parser.add_argument("reps", type=int, help="Number of reps")
    parser.add_argument("itr", type=int, help="Iteration number")
    parser.add_argument("temp", type=float, help="Simulation temperature")
    parser.add_argument("--varis", nargs="+", type=str, help="Simulation variants")

    return parser.parse_args()


if __name__ == "__main__":
    main()
