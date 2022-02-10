"""Functions for calculating analytical results"""


import math

import scipy.constants as constants


def calc_I_circle(R_filament):
    return math.pi * R_filament**4 / 4


def calc_I_square(height):
    return height**4 / 12


def calc_youngs_modulus(R_filament, EI):
    I_circle = calc_I_circle(R_filament)

    return EI / I_circle


def calc_max_radius(length, n_filaments):
    return n_filaments * length / (2 * math.pi)


def calc_min_radius(max_radius):
    return max_radius / 2


def calc_cuboid_ring_energy(length, height, R_ring, EI):
    """Assumes that height and width are equal"""

    # I could make the cuboid approximation more accurate than assuming the
    # height of the beam is twice the radius
    R_filament = height / 2
    E = calc_youngs_modulus(R_filament, EI)
    I_square = calc_I_square(height)

    return E * I_square * length / (2 * R_ring**2)


def calc_sliding_force(params):
    ks = params["ks"]
    kd = params["kd"]
    T = params["T"]
    delta = params["delta"]
    Xc = params["Xc"]

    return constants.k * T / delta * math.log(1 + ks**2 * Xc / (kd * (ks + Xc) ** 2))


def calc_sliding_energy(overlap_L, params):
    return -overlap_L * calc_sliding_force(params)


def calc_bending_force(R_ring, params):
    return params["EI"] * params["Lf"] / R_ring**3


def calc_bending_energy(R_ring, params):
    return params["EI"] * params["Lf"] / (2 * R_ring**2)
