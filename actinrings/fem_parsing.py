"""Functions for parsing output from FEM calculations"""


import itertools

import pandas as pd


def collect_integrated_data(filename_template, params):

    # Convert dictionary to lists of tuples of key value for key's list
    paramsl = []
    for key, value in params.items():
        paramsl.append([])
        for v in value:
            paramsl[-1].append((key, v))

    frames = []
    for combo in itertools.product(*paramsl):

        # Convert tuple of tuples to a dictionary
        format_dic = {}
        for pair in combo:
            format_dic[pair[0]] = pair[1]

        filename = filename_template.format(**format_dic)
        try:
            frame = pd.read_csv(filename, sep=' ')
            for pair in combo:
                if pair[0] not in frame.columns:
                    frame[pair[0]] = pair[1]

            frames.append(frame)

        except FileNotFoundError:
            print('Skipping {}'.format(filename))

    return pd.concat(frames)


def reduce_data(data, reduce_variables):
    for key, value in reduce_variables.items():
        data = data[data[key] == value]

    return data
