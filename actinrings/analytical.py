"""Functions for calculating analytical results"""


import math


# From mameren2009
EI = 7.1e-26


def calc_I_circle(R_filament):
    return math.pi*R_filament**4/4


def calc_I_square(height):
    return height**4/12


def calc_youngs_modulus(R_filament):
    I_circle = calc_I_circle(R_filament)

    return EI/I_circle


def calc_beam_energy_cuboid_ring(length, height, R_ring):
    """Assumes that height and width are equal"""

    # I could make the cuboid approximation more accurate than assuming the
    # height of the beam is twice the radius
    R_filament = height/2
    E = calc_youngs_modulus(R_filament)
    I_square = calc_I_square(height)

    return 3*E*I_square*length/(4*R_ring**2)


def calc_wlc_energy_cuboid_ring(length, height, R_ring):
    """Assumes that height and width are equal"""

    # I could make the cuboid approximation more accurate than assuming the
    # height of the beam is twice the radius
    R_filament = height/2
    E = calc_youngs_modulus(R_filament)
    I_square = calc_I_square(height)

    return E*I_square*length/(2*R_ring**2)
