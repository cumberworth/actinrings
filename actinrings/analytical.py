"""Functions for calculating analytical results.

The params dictionary has a standard set of keys. Only a subset of keys may need to be
defined for a particular method. Possible keys:
        T: Temperature / K
        delta: Lattice spacing / m
        Lf: Filament length / m
        Xc: Crosslinker concentration / M
        ks: Equilibrium constant for single crosslinker binding
        kd: Equilibrium constant for double crosslinker binding
        EI: Bending rigidity / N m^2
"""


import math

import numpy as np
from scipy import constants
from scipy import optimize


def calc_I_circle(R: float) -> float:
    """Calculate the second moment of area of a circle centered on an axis."""
    return math.pi * R**4 / 4


def calc_I_square(height: float) -> float:
    """Calculate the second moment of square centered on and level with an axis."""
    return height**4 / 12


def calc_youngs_modulus(R_filament: float, EI: float) -> float:
    """Calculate Young's modulus of a filament.

    Given the bending rigidity and assuming a circular cross section, calculate the
    Young's modulus.

    Args:
        R_filament: Radius of the filament / m
        EI: Bending rigidity / N m^-2
    """
    I_circle = calc_I_circle(R_filament)

    return EI / I_circle


def calc_max_radius(length: float, Nsca: int) -> float:
    """Calculate the maximum possible radius for a ring.

    This occurs when the scaffold filaments have no overlaps.

    Args:
        length: Filament length / m
        Nsca: Number of scaffold filaments
    """
    return Nsca * length / (2 * math.pi)


def calc_min_radius(max_radius: float) -> float:
    """Calculate the minimum possible radius for a ring."""
    return max_radius / 2


def calc_sliding_force(params: dict) -> float:
    """Calculate the sliding force for a single overlap.

    Args:
        params: System parameters
    """
    ks = params["ks"]
    kd = params["kd"]
    T = params["T"]
    delta = params["delta"]
    Xc = params["Xc"]

    return constants.k * T / delta * math.log(1 + ks**2 * Xc / (kd * (ks + Xc) ** 2))


def calc_sliding_energy(overlap_L: float, params: dict) -> float:
    """Calculate the sliding energy for given overlap length.

    Args:
        overlap_L: Total length of overlap(s) / m
        params: System parameters
    """
    return -overlap_L * calc_sliding_force(params)


def calc_bending_force(R_ring: float, params: dict) -> float:
    """Calculate the elastic bending force of a filament in a ring.

    Args:
        R_ring: Radius of the ring / m
        params: System parameters
    """
    return params["EI"] * params["Lf"] / R_ring**3


def calc_bending_energy(R_ring: float, params: dict) -> float:
    """Calculate the elastic bending energy of a filament in a ring.

    Args:
        R_ring: Radius of the ring / m
        params: System parameters
    """
    return params["EI"] * params["Lf"] / (2 * R_ring**2)


def calc_ring_energy(R_ring: float, N: int, Nsca: int, params: dict) -> float:
    """Calculate the total ring energy.

    Args:
        R_ring: Radius of the ring / m
        N: Total number of filaments in ring
        Nsca: Number of scaffold filaments
        params: System parameters
    """
    R_ring_max = calc_max_radius(params["Lf"], Nsca)
    overlap_L = 2 * math.pi * (R_ring_max - R_ring) / Nsca
    overlaps = Nsca + 2 * (N - Nsca)
    sliding_energy = overlaps * calc_sliding_energy(overlap_L, params)
    bending_energy = N * calc_bending_energy(R_ring, params)
    total_energy = sliding_energy + bending_energy

    return total_energy


def calc_ring_bending_energy(R_ring: float, N: int, params: dict) -> float:
    """Calculate the total elastic bending energy of all filaments in a ring.

    Args:
        R_ring: Radius of the ring / m
        N: Total number of filaments in ring
        params: System parameters
    """
    return N * calc_bending_energy(R_ring, params)


def calc_ring_sliding_energy(R_ring: float, N: int, Nsca: int, params: dict) -> float:
    """Calculate the total sliding energy of all filaments in a ring.

    Args:
        R_ring: Radius of the ring / m
        N: Total number of filaments in ring
        Nsca: Number of scaffold filaments
        params: System parameters
    """
    R_ring_max = calc_max_radius(params["Lf"], Nsca)
    overlap_L = 2 * math.pi * (R_ring_max - R_ring) / Nsca
    overlaps = Nsca + 2 * (N - Nsca)

    return overlaps * calc_sliding_energy(overlap_L, params)


def calc_ring_force(R_ring: float, N: int, Nsca: int, params: dict) -> float:
    """Calculate the total radial force of all filaments in a ring.

    Args:
        R_ring: Radius of the ring / m
        N: Total number of filaments in ring
        Nsca: Number of scaffold filaments
        params: System parameters
    """
    ks = params["ks"]
    kd = params["kd"]
    T = params["T"]
    delta = params["delta"]
    Xc = params["Xc"]
    EI = params["EI"]
    Lf = params["Lf"]
    sliding_force = -(
        2
        * math.pi
        * constants.k
        * T
        * (2 * N - Nsca)
        * math.log(1 + ks**2 * Xc / (kd * (ks + Xc) ** 2))
        / (delta * Nsca)
    )
    bending_force = EI * N * Lf / R_ring**3
    total_force = sliding_force + bending_force

    return total_force


def calc_equilibrium_ring_radius(N: int, Nsca: int, params: dict) -> float:
    """Calculate the equilibrium radius of a ring analytically.

    Args:
        N: Total number of filaments in ring
        Nsca: Number of scaffold filaments
        params: System parameters
    """
    ks = params["ks"]
    kd = params["kd"]
    T = params["T"]
    delta = params["delta"]
    Xc = params["Xc"]
    EI = params["EI"]
    Lf = params["Lf"]
    num = EI * N * delta * Lf * Nsca
    denom = (
        2
        * math.pi
        * T
        * constants.k
        * math.log(1 + ks**2 * Xc / (kd * (ks + Xc) ** 2))
        * (2 * N - Nsca)
    )

    return (num / denom) ** (1 / 3)


def calc_equilibrium_radius_numerical(N: int, Nsca: int, params: dict) -> float:
    """Calculate the equilibrium radius of a ring numerically by minimizing energy.

    Args:
        N: Total number of filaments in ring
        Nsca: Number of scaffold filaments
        params: System parameters
    """
    max_radius = calc_max_radius(params["Lf"], Nsca)
    min_radius = max_radius / 2
    res = optimize.minimize_scalar(
        calc_ring_energy,
        method="bounded",
        bounds=(1e-30, 2 * max_radius),
        args=(N, Nsca, params),
        options={"xatol": 1e-12},
    )

    radius = res.x
    if radius > max_radius:
        print("Ring will fall apart under these conditions")
        print(f"Max radius {max_radius}, calculated radius: {radius}")
        raise

    elif radius < min_radius:
        print("Ring will violate overlap assumptions under these conditions")
        print(f"Min radius {min_radius}, calculated radius: {radius}")
        raise

    return res.x


def calc_degeneracies(heights: list, lf: int, N: int, include_height=False):
    """For testing purposes."""
    max_height = 2 * lf - 3
    degens = []
    for h in heights:
        overlap = max_height - h + 2
        if include_height:
            if N == 2:
                degens.append((h + 1) * (overlap - 1))
            else:
                degens.append((h + 1) * (overlap - 1) * overlap ** (N - 2))
        else:
            if N == 2:
                degens.append(overlap - 1)
            else:
                degens.append((overlap - 1) * overlap ** (N - 2))

    return np.array(degens)


def calc_koff(koff, F, xbeta, T):
    """Calculate the force-dependent off rate.

    This assumes a pulling force so just uses the magnitude.

    Args:
        koff: Off rate with no force / s^-1
        F: Applied force / N
        xbeta: Distance from well to transition state
        T: Temerature
    """
    return koff * np.exp(abs(F) * xbeta / (constants.k * T))
