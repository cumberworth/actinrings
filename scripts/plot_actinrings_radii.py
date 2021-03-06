#!/usr/bin/env python

"""Plot radii at each time step.

Plots a single replicate of the given iteration for a run. The itr should be that which
is in the file name, rather than the row entry in the output (these can differ when the
run has been continued from a previous).
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from actinrings import plots


def main():
    args = vars(parse_args())
    p = plots.RadiiPlot(args)
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
    parser.add_argument("vari", type=str, help="Variant name")
    parser.add_argument("rep", type=int, help="Replicate number")
    parser.add_argument("itr", type=int, help="Iteration number")

    return parser.parse_args()


if __name__ == "__main__":
    main()
