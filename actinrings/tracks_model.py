"""Functions for the tracks model"""

import math

from scipy import constants
from scipy import optimize

from actinrings import analytical


def calc_ring_energy(R_ring, N, min_N, params):
    R_ring_max = analytical.calc_max_radius(params['Lf'], min_N)
    overlap_L = 2*math.pi*(R_ring_max - R_ring)/min_N
    overlaps = min_N + 2*(N - min_N)
    sliding_energy = overlaps*analytical.calc_sliding_energy(overlap_L, params)
    bending_energy = N*analytical.calc_bending_energy(R_ring, params)
    total_energy = sliding_energy + bending_energy

    return total_energy


def calc_ring_force(R_ring, N, min_N, params):
    sliding_force = (2*math.pi*constants.k*params['T']*(2*N - min_N) *
                     math.log(params['Xc'] / params['kd'] + 1) /
                     (params['delta']*min_N))
    bending_force = -params['EI']*N*params['Lf']/R_ring**3
    total_force = sliding_force + bending_force

    return total_force


def calc_equilibrium_ring_radius(N, min_N, params):
    num = params['EI']*N*params['delta']*params['Lf']*min_N
    denom = (2*math.pi*params['T']*constants.k *
             math.log(params['Xc']/params['kd'] + 1)*(2*N - min_N))

    return (num/denom)**(1/3)


def calc_equilibrium_radius_numerical(N, min_N, params):
    max_radius = analytical.calc_max_radius(params['Lf'], min_N)
    min_radius = max_radius/2
    res = optimize.minimize_scalar(calc_ring_energy,
                                   method='bounded',
                                   bounds=(1e-30, 2*max_radius),
                                   args=(N, min_N, params),
                                   options={'xatol': 1e-12})

    radius = res.x
    if (radius > max_radius):
        print('Ring will fall apart under these conditions')
        print(f'Max radius {max_radius}, calculated radius: {radius}')
        raise

    elif (radius < min_radius):
        print('Ring will violate overlap assumptions under these conditions')
        print(f'Min radius {min_radius}, calculated radius: {radius}')
        raise

    return res.x
