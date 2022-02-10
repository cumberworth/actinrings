"""Functions for the double overlap model"""


import math

from scipy import optimize
from scipy import constants

from actinrings import analytical


def calc_equilibrium_radius(N, params):
    num = params["EI"] * N * params["filament_L"] * params["delta"]
    denom = (
        2
        * math.pi
        * constants.k
        * params["T"]
        * math.log(1 + params["X_C"] / params["kd"])
    )

    return (num / denom) ** (1 / 3)


def calc_equilibrium_concentration(N, R_ring, params):
    num = params["EI"] * N * params["filament_L"] * params["delta"]
    denom = 2 * math.pi * R_ring**3 * constants.k * params["T"]

    return params["kd"] * (math.exp(num / denom) - 1)


def calc_ring_energy(R_ring, N, params):
    R_ring_max = analytical.calc_max_radius(params["filament_L"], N)
    overlap_L = 2 * math.pi * (R_ring_max - R_ring) / N
    sliding_energy = N * analytical.calc_sliding_energy(overlap_L, params)
    bending_energy = N * analytical.calc_bending_energy(R_ring, params)
    total_energy = sliding_energy + bending_energy

    return total_energy


def calc_equilibrium_radius_numerical(N, params):
    max_radius = analytical.calc_max_radius(params["filament_L"], N)
    res = optimize.minimize_scalar(
        calc_ring_energy,
        method="bounded",
        bounds=(1e-30, 2 * max_radius),
        args=(N, params),
        options={"xatol": 1e-12},
    )

    return res.x
