"""Functions for parsing output from FEM calculations"""


import pandas as pd


def parse_integrated_output(filename):
    """Read integrated results and return a dataframe"""

    return pd.read_csv(filename, sep=' ')


def concatenate_integrated_outputs(frames):
    """Concatenate a list of dataframes"""

    return pd.concat(frames)
