"""Plotting functions and classes"""

import json

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
import numpy as np
import pandas as pd
from scipy import constants

from actinrings import analytical
from actinrings import tracks_model
from matplotlibstyles import styles


def setup_figure(w=8.6, h=6):
    styles.set_thin_latex_style()
    figsize = (styles.cm_to_inches(w), styles.cm_to_inches(h))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def save_figure(f, plot_filebase):
    f.savefig(plot_filebase + ".pgf", transparent=True)
    f.savefig(plot_filebase + ".pdf", transparent=True)
    f.savefig(plot_filebase + ".png", transparent=True)


def set_line_labels_by_pos(
    line, ax, xpos, ha, va, ypos=None, xshift=0, yshift=0, alpha=1
):
    xdata = line.get_xdata()
    if ypos is None:
        ypos = line.get_ydata()[np.abs(xdata - xpos).argmin()]

    ax.text(
        xpos + xshift,
        ypos + yshift,
        line.get_label(),
        color=line.get_color(),
        horizontalalignment=ha,
        verticalalignment=va,
        alpha=alpha,
    )


def set_line_labels_by_index(line, ax, index, ha, va, xshift=0, yshift=0, alpha=1):
    xpos = line.get_xdata()[index]
    ypos = line.get_ydata()[index]
    ax.text(
        xpos + xshift,
        ypos + yshift,
        line.get_label(),
        color=line.get_color(),
        horizontalalignment=ha,
        verticalalignment=va,
        alpha=alpha,
    )


def set_line_labels_to_middle(line, ax, ha, va, xshift=0, yshift=0, alpha=1):
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    middle_index = (np.argmax(ydata) + np.argmin(ydata)) // 2
    xpos = xdata[middle_index]
    ypos = ydata[middle_index]
    ax.text(
        xpos,
        ypos + yshift,
        line.get_label(),
        color=line.get_color(),
        horizontalalignment=ha,
        verticalalignment=va,
        alpha=alpha,
    )


def load_sim_params(args, filename):
    with open(filename) as file:
        parms = json.load(file)

    parms["N"] = parms["Nfil"]
    parms["Nmin"] = parms["Nsca"]
    parms["temp"] = parms["T"]

    return args | parms


class Plot:
    def __init__(self, args):
        self._args = args

    def set_labels(self, ax):
        plt.legend()


class FreqsPlot(Plot):
    def plot_figure(self, f, ax):
        itr = self._args["itr"]
        input_dir = self._args["input_dir"]
        for vari in self._args["varis"]:
            for rep in range(1, self._args["reps"] + 1):
                filename = f"{input_dir}/{vari}/{vari}_rep-{rep}.freqs"
                freqs = pd.read_csv(filename, header=0, delim_whitespace=True)
                heights = freqs.columns.astype(int)

                # ax.plot(heights, np.log10(freqs.iloc[itr - 1]))
                ax.plot(heights, freqs.iloc[itr - 1])

    def setup_axis(self, ax):
        ax.set_ylabel(r"Fraction")
        ax.set_xlabel(r"Lattice height")


class LFEsPlot(Plot):
    def plot_figure(self, f, ax):
        itr = self._args["itr"]
        input_dir = self._args["input_dir"]
        for vari in self._args["varis"]:
            for rep in range(1, self._args["reps"] + 1):
                filename = f"{input_dir}/{vari}/{vari}_rep-{rep}.biases"
                biases = pd.read_csv(filename, header=0, delim_whitespace=True)
                heights = biases.columns.astype(int)
                lfes = -biases / (self._args["temp"] * constants.k)

                ax.plot(heights, lfes.iloc[itr - 1], marker="o")

    def setup_axis(self, ax):
        ax.set_ylabel(r"$k_\mathrm{b}T$")
        ax.set_xlabel("Lattice height")


class RadiusLFEsAnalyticalPlot(Plot):
    def plot_figure(self, f, ax, calc_degen=False):
        itr = self._args["itr"]
        input_dir = self._args["input_dir"]
        vari = self._args["vari"]
        delta = self._args["delta"]
        Lf = self._args["Lf"]
        Nmin = self._args["Nmin"]
        N = self._args["N"]
        temp = self._args["temp"]
        lf = self._args["lf"]

        cmap = cm.get_cmap("tab10")

        filename = f"{input_dir}/{vari}/{vari}_rep-1.biases"
        biases = pd.read_csv(filename, header=0, delim_whitespace=True)
        heights = biases.columns.astype(int)
        radii = (heights + 1) * delta / (2 * np.pi)
        radii_scaled = radii / 1e-6

        align_i = -1

        for rep in range(1, self._args["reps"] + 1):
            filename = f"{input_dir}/{vari}/{vari}_rep-{rep}.biases"
            biases = pd.read_csv(filename, header=0, delim_whitespace=True)
            heights = biases.columns.astype(int)
            lfes = -biases / (temp * constants.k)
            lfes_itr = lfes.iloc[itr - 1]
            # lfes_itr = lfes_itr + np.log(heights + 1)
            lfes_itr -= lfes_itr[align_i]
            ax.plot(radii_scaled, lfes_itr, color=cmap(0))

        energies = [
            tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
        ]
        energies = np.array(energies)
        energies_scaled = energies / (constants.k * temp)
        energies_scaled -= energies_scaled[align_i]

        ax.plot(radii_scaled, energies_scaled, color=cmap(1))

        if calc_degen:
            # degens = tracks_model.calc_degeneracies(heights, lf, include_height=True)
            # boltz_weights = degens*np.exp(-energies/constants.k/temp)
            # lfes_anal = -np.log(boltz_weights / boltz_weights[align_i])

            # ax.plot(radii_scaled, lfes_anal, color=cmap(2))

            degens = tracks_model.calc_degeneracies(
                heights, lf, N, include_height=False
            )
            boltz_weights = degens * np.exp(-energies / constants.k / temp)
            lfes_anal = -np.log(boltz_weights / boltz_weights[align_i])

            ax.plot(radii_scaled, lfes_anal, color=cmap(2))

    def setup_axis(self, ax):
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$LFE / k_\mathrm{b}T$")


class RadiusLFEsNPlot(Plot):
    def plot_figure(self, f, ax, alpha=1, calc_anal=False, align=False):
        itrs = self._args["itrs"]
        input_dir = self._args["input_dir"]
        varis = self._args["varis"]
        delta = self._args["delta"]
        Lf = self._args["Lf"]
        Nmin = self._args["Nmin"]
        Ns = self._args["Ns"]
        temp = self._args["temp"]
        lf = self._args["lf"]

        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])

        # Get heights (same for all N)
        filename = f"{input_dir}/{varis[0]}/{varis[0]}_rep-1.biases"
        biases = pd.read_csv(filename, header=0, delim_whitespace=True)
        heights = biases.columns.astype(int)
        radii = (heights + 1) * delta / (2 * np.pi)
        radii_scaled = radii / 1e-6

        align_i = -1

        radii_scaled = radii / 1e-6
        N_lfes = []
        min_lfes = []
        min_radii = []
        for N, vari, itr in zip(Ns, varis, itrs):
            if calc_anal:
                lfes = [
                    tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
                ]
                lfes = np.array(lfes)
                lfes_scaled = lfes / (constants.k * temp)
                if align:
                    lfes_scaled -= lfes_scaled[align_i]

                N_lfes.append(lfes_scaled)
                min_lfe_i = np.argmin(lfes_scaled)
                min_lfes.append(lfes_scaled[min_lfe_i])
                min_radii.append(np.min(radii_scaled[min_lfe_i]))
            else:
                N_lfes.append([])
                min_radii.append([])
                min_lfes.append([])
                for rep in range(1, self._args["reps"] + 1):
                    filename = f"{input_dir}/{vari}/{vari}_rep-{rep}.biases"
                    biases = pd.read_csv(filename, header=0, delim_whitespace=True)
                    heights = biases.columns.astype(int)
                    lfes = -biases / (temp * constants.k)
                    lfes_itr = lfes.iloc[itr - 1]
                    if align:
                        lfes_itr -= lfes_itr[align_i]

                    N_lfes[-1].append(lfes_itr)

                    min_lfe_i = np.argmin(lfes_itr)
                    min_lfes[-1].append(lfes_itr[min_lfe_i])
                    min_radii[-1].append(np.min(radii_scaled[min_lfe_i]))

        # Scale energy to have min at 0
        # N_lfes = np.array(N_lfes)
        # min_lfe = np.min(min_lfes)
        # N_lfes -= min_lfe
        # min_lfes -= min_lfe

        # Plot
        for i, N in enumerate(Ns):
            label = rf"$N_\text{{f}}={N}$"
            if calc_anal:
                label = label + ",\nanalytical"

            if calc_anal:
                lfes = N_lfes[i]
                ax.plot(
                    radii_scaled,
                    lfes,
                    color=mappable.to_rgba(N),
                    label=label,
                    alpha=alpha,
                )
            else:
                for rep in range(self._args["reps"]):
                    lfes = N_lfes[i][rep]
                    ax.plot(
                        radii_scaled,
                        lfes,
                        color=mappable.to_rgba(N),
                        label=label,
                        alpha=alpha,
                        marker=".",
                        markersize=1,
                    )

        ax.plot(
            np.mean(min_radii, axis=1),
            np.mean(min_lfes, axis=1),
            linestyle="None",
            marker="*",
            markersize=4,
            markeredgewidth=0,
            alpha=alpha,
        )

    def setup_axis(self, ax):
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$LFE / k_\mathrm{b}T$")


class RadiusForcesAnalyticalPlot(Plot):
    def plot_figure(self, f, ax, calc_degen=False):
        itr = self._args["itr"]
        input_dir = self._args["input_dir"]
        vari = self._args["vari"]
        delta = self._args["delta"]
        Lf = self._args["Lf"]
        Nmin = self._args["Nmin"]
        N = self._args["N"]
        temp = self._args["temp"]
        lf = self._args["lf"]

        cmap = cm.get_cmap("tab10")

        filename = f"{input_dir}/{vari}/{vari}_rep-1.biases"
        biases = pd.read_csv(filename, header=0, delim_whitespace=True)
        heights = biases.columns.astype(int)
        radii = (heights + 1) * delta / (2 * np.pi)
        forces_radii = radii[1:] - delta
        radii_scaled = radii / 1e-6
        forces_radii_scaled = forces_radii / 1e-6

        for rep in range(1, self._args["reps"] + 1):
            filename = f"{input_dir}/{vari}/{vari}_rep-{rep}.biases"
            biases = pd.read_csv(filename, header=0, delim_whitespace=True)
            heights = biases.columns.astype(int)
            lfes = -biases
            lfes_itr = lfes.iloc[itr - 1]
            forces = -np.diff(lfes_itr) / (delta / (2 * np.pi))
            forces_scaled = np.array(forces) / 1e-12
            ax.plot(forces_radii_scaled, forces_scaled, color=cmap(0))

        a_forces = [tracks_model.calc_ring_force(r, N, Nmin, self._args) for r in radii]
        a_forces_scaled = np.array(a_forces) / 1e-12

        ax.plot(radii_scaled, a_forces_scaled, color=cmap(1))

        if calc_degen:
            energies = [
                tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
            ]
            energies = np.array(energies)

            # degens = tracks_model.calc_degeneracies(heights, lf, include_height=True)
            # boltz_weights = degens*np.exp(-energies/constants.k/temp)
            # lfes_anal = -constants.k*temp*np.log(boltz_weights)
            # forces_anal = -np.diff(lfes_anal)/(delta / (2*np.pi))
            # forces_anal_scaled = np.array(forces_anal) / 1e-12

            # ax.plot(forces_radii_scaled, forces_anal_scaled, color=cmap(2))

            degens = tracks_model.calc_degeneracies(
                heights, lf, N, include_height=False
            )
            boltz_weights = degens * np.exp(-energies / constants.k / temp)
            lfes_anal = -constants.k * temp * np.log(boltz_weights)
            forces_anal = -np.diff(lfes_anal) / (delta / (2 * np.pi))
            forces_anal_scaled = np.array(forces_anal) / 1e-12

            ax.plot(forces_radii_scaled, forces_anal_scaled, color=cmap(2))

    def setup_axis(self, ax):
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$F / \si{\pico\newton}$")


class RadiusForcesNPlot(Plot):
    def plot_figure(self, f, ax, calc_anal=False, alpha=1, zero_line=False):
        itrs = self._args["itrs"]
        input_dir = self._args["input_dir"]
        varis = self._args["varis"]
        delta = self._args["delta"]
        Lf = self._args["Lf"]
        Nmin = self._args["Nmin"]
        Ns = self._args["Ns"]
        temp = self._args["temp"]
        lf = self._args["lf"]

        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])

        # Get heights (same for all N)
        filename = f"{input_dir}/{varis[0]}/{varis[0]}_rep-1.biases"
        biases = pd.read_csv(filename, header=0, delim_whitespace=True)
        heights = biases.columns.astype(int)
        radii = (heights + 1) * delta / (2 * np.pi)
        forces_radii = radii[1:] - delta
        radii_scaled = radii / 1e-6
        forces_radii_scaled = forces_radii / 1e-6

        for N, vari, itr in zip(Ns, varis, itrs):
            if calc_anal:
                forces = [
                    tracks_model.calc_ring_force(r, N, Nmin, self._args)
                    for r in forces_radii
                ]
                forces_scaled = np.array(forces) / 1e-12
                label = rf"$N_\text{{f}}={N}$"
                label = label + ",\nanalytical"
                ax.plot(
                    forces_radii_scaled,
                    forces_scaled,
                    color=mappable.to_rgba(N),
                    label=label,
                    alpha=alpha,
                )
            else:
                for rep in range(1, self._args["reps"] + 1):
                    filename = f"{input_dir}/{vari}/{vari}_rep-{rep}.biases"
                    biases = pd.read_csv(filename, header=0, delim_whitespace=True)
                    heights = biases.columns.astype(int)
                    lfes = -biases
                    lfes_itr = lfes.iloc[itr - 1]
                    forces = -np.diff(lfes_itr) / (delta / (2 * np.pi))
                    forces_scaled = np.array(forces) / 1e-12
                    label = rf"$N_\text{{f}}={N}$"
                    ax.plot(
                        forces_radii_scaled,
                        forces_scaled,
                        color=mappable.to_rgba(N),
                        label=label,
                        alpha=alpha,
                        marker=".",
                        markersize=1,
                    )

        if zero_line:
            ax.axhline(0, linestyle="dashed")

    def setup_axis(self, ax):
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$F / \si{\pico\newton}$")


class RadiiPlot(Plot):
    def plot_figure(self, f, ax):
        vari = self._args["vari"]
        input_dir = self._args["input_dir"]
        rep = self._args["rep"]
        itr = self._args["itr"]
        filename = f"{input_dir}/{vari}/{vari}_rep-{rep}_iter-{itr}.ops"
        ops = pd.read_csv(filename, header=0, delim_whitespace=True)
        ax.plot(ops["step"], ops["radius"])

    def setup_axis(self, ax):
        ax.set_ylabel(r"Radius")
        ax.set_xlabel(r"Step")


class LfEradiusNPlot(Plot):
    def plot_figure(self, f, ax):
        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        Lfs = self._args["Lfs"]
        Ns = self._args["Ns"]
        Nmin = self._args["Nmin"]
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])
        xvals = []
        yvals = []
        for N in Ns:
            radii = []
            lengths = []
            for j, Lf in enumerate(Lfs):
                self._args["Lf"] = Lf
                max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
                min_radius = analytical.calc_min_radius(max_radius)
                r = tracks_model.calc_equilibrium_ring_radius(N, Nmin, self._args)
                if r >= min_radius and r <= max_radius:
                    radii.append(r)
                    lengths.append(Lfs[j])

            # Convert units
            radii = np.array(radii) / 1e-6
            lengths = np.array(lengths) / 1e-6

            # Plot
            label = rf"$N_\text{{f}}={N}"
            ax.plot(lengths, radii, color=mappable.to_rgba(N), label=label)

    def setup_axis(self, ax):
        ax.set_title(rf'$N_\text{{sca}} = {self._args["Nmin"]}$', loc="center")
        ax.set_xlabel(r"$L^\text{f} / \si{\micro\meter}$")
        ax.set_ylabel(r"$R_\text{eq} / \si{\micro\meter}$")


class LfEradiusNminPlot(Plot):
    def plot_figure(self, f, ax):
        Nmins = self._args["Nmins"]
        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Nmins[0], Nmins[-1])
        N = self._args["N"]
        Lfs = self._args["Lfs"]
        for Nmin in Nmins:
            radii = []
            lengths = []
            for j, Lf in enumerate(Lfs):
                self._args["Lf"] = Lf
                max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
                min_radius = analytical.calc_min_radius(max_radius)
                r = tracks_model.calc_equilibrium_ring_radius(N, Nmin, self._args)
                if r >= min_radius and r <= max_radius:
                    radii.append(r)
                    lengths.append(Lfs[j])

            # Convert units
            radii = np.array(radii) / 1e-6
            lengths = np.array(lengths) / 1e-6

            # Plot
            label = rf"$N_\text{{sca}}={Nmin}"
            ax.plot(lengths, radii, color=mappable.to_rgba(Nmin), label=label)

    def setup_axis(self, ax):
        ax.set_title(rf'$N_\text{{f}} = {self._args["N"]}$', loc="center")
        ax.set_xlabel(r"$L^\text{f} / \si{\micro\meter}$")
        ax.set_ylabel(r"$R_\text{eq} / \si{\micro\meter}$")


class XcForcePlot(Plot):
    def plot_figure(self, f, ax):
        fractions = self._args["fractions"]
        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, fractions[0], fractions[-1])
        N = self._args["N"]
        Nmin = self._args["Nmin"]
        Xcs = self._args["Xcs"]
        Lf = self._args["Lf"]
        max_radius = analytical.calc_max_radius(Lf, Nmin)
        for fraction in fractions:
            r = max_radius * fraction
            forces = []
            for Xc in Xcs:
                self._args["Xc"] = Xc
                force = tracks_model.calc_ring_force(r, N, Nmin, self._args)
                forces.append(force)

            # Convert units
            forces_scaled = np.array(forces) / 1e-12

            # Plot
            if fraction == 1:
                label = r"$R_{\text{max}}$"
            else:
                label = rf"${fraction} R_{{\text{{max}}}}$"

            ax.plot(Xcs, forces_scaled, color=mappable.to_rgba(fraction), label=label)

        ax.axhline(0, linestyle="dashed")

    def setup_axis(self, ax):
        Lf_scaled = self._args["Lf"] / 1e-6
        N = self._args["N"]
        Nmin = self._args["Nmin"]
        ax.set_title(
            rf"$N_\text{{sca}} = {Nmin}$, $N_\text{{f}} = {N}$, $L_\text{{f}}"
            rf"= \SI{{{int(Lf_scaled)}}}{{\micro\meter}}$",
            loc="center",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\text{[X]} / \si{\molar}$")
        ax.set_ylabel(r"$F / \si{\pico\newton}$")
        ax.set_ylim(top=3)
        minor_ticks = ticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9))
        ax.xaxis.set_minor_locator(minor_ticks)
        # ax.set_yscale('log')
        # minor_ticks = ticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9))
        # ax.yaxis.set_minor_locator(minor_ticks)


class RadiusEnergyLfPlot(Plot):
    def plot_figure(self, f, ax, calc_degens=False, alpha=1):
        Nmin = self._args["Nmin"]
        N = self._args["N"]
        Lfs = self._args["Lfs"]
        delta = self._args["delta"]
        temp = self._args["T"]

        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Lfs[0], Lfs[-1])
        energies = []
        radiis = []
        min_energies = []
        min_radii = []
        for i, Lf in enumerate(Lfs):
            self._args["Lf"] = Lf
            if calc_degens:
                lf = self._args["lfs"][i]
                max_height = int(Nmin * lf - Nmin - 1)
                min_height = int((Nmin // 2) * lf - 1)
                heights = range(min_height, max_height + 1)
                heights = np.array(heights)
                radii = (heights + 1) * delta / (2 * np.pi)
            else:
                samples = self._args["samples"]
                max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
                min_radius = analytical.calc_min_radius(max_radius)
                radii = np.linspace(min_radius, max_radius, num=samples)

            radii_scaled = radii / 1e-6
            radiis.append(radii_scaled)
            energy = [
                tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
            ]
            energy = np.array(energy)
            if calc_degens:
                degens = tracks_model.calc_degeneracies(
                    heights, lf, N, include_height=False
                )
                boltz_weights = degens * np.exp(-energy / constants.k / temp)
                energy = -constants.k * temp * np.log(boltz_weights)

            energy_scaled = np.array(energy) / (constants.k * self._args["T"])
            energies.append(energy_scaled)
            min_energy_i = np.argmin(energy_scaled)
            min_energies.append(energy_scaled[min_energy_i])
            min_radii.append(np.min(radii_scaled[min_energy_i]))

        # Scale energy to have min at 0
        # energies = np.array(energies)
        # min_energy = np.min(min_energies)
        # energies -= min_energy
        # min_energies -= min_energy

        # Plot
        for radii, Lf, energy in zip(radiis, Lfs, energies):
            Lf_scaled = np.round(Lf / 1e-6, decimals=1)
            label = rf"$L_\text{{f}}=\SI{{{Lf_scaled}}}{{\micro\meter}}$"
            ax.plot(radii, energy, color=mappable.to_rgba(Lf), label=label, alpha=alpha)

        ax.plot(
            min_radii,
            min_energies,
            linestyle="None",
            marker="*",
            markeredgewidth=0,
            alpha=alpha,
        )

    def setup_axis(self, ax):
        Nmin = self._args["Nmin"]
        N = self._args["N"]
        ax.set_title(rf"$N_\text{{sca}} = {Nmin}$, $N_\text{{f}} = {N}$", loc="center")
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$\upDelta \Phi / \si{\kb} T$ \hspace{7pt}", labelpad=-4)

    #    ax.set_ylim(top=30)
    #    ax.set_xlim(left=3, right=8.7)


class RadiusEnergyNPlot(Plot):
    def plot_figure(self, f, ax, calc_degens=False, alpha=1):
        Ns = self._args["Ns"]
        Nmin = self._args["Nmin"]
        temp = self._args["T"]
        delta = self._args["delta"]

        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])
        max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
        min_radius = analytical.calc_min_radius(max_radius)
        if calc_degens:
            lf = self._args["lf"]
            max_height = Nmin * lf - Nmin - 1
            min_height = (Nmin // 2) * lf - 1
            heights = range(min_height, max_height + 1)
            heights = np.array(heights)
            radii = (heights + 1) * delta / (2 * np.pi)
        else:
            samples = self._args["samples"]
            radii = np.linspace(min_radius, max_radius, num=samples)

        radii_scaled = radii / 1e-6
        energies = []
        min_energies = []
        min_radii = []
        for N in Ns:
            energy = [
                tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
            ]
            energy = np.array(energy)
            if calc_degens:
                degens = tracks_model.calc_degeneracies(
                    heights, lf, N, include_height=False
                )
                boltz_weights = degens * np.exp(-energy / constants.k / temp)
                energy = -constants.k * temp * np.log(boltz_weights)

            energy_scaled = np.array(energy) / (constants.k * temp)
            energies.append(energy_scaled)
            min_energy_i = np.argmin(energy_scaled)
            min_energies.append(energy_scaled[min_energy_i])
            min_radii.append(np.min(radii_scaled[min_energy_i]))

        # Scale energy to have min at 0
        # energies = np.array(energies)
        # min_energy = np.min(min_energies)
        # energies -= min_energy
        # min_energies -= min_energy

        # Plot
        for N, energy in zip(Ns, energies):
            label = rf"$N_\text{{f}}={N}$"
            if calc_degens:
                label = label + ", with degenarcy"

            ax.plot(
                radii_scaled,
                energy,
                color=mappable.to_rgba(N),
                label=label,
                alpha=alpha,
            )

        ax.plot(
            min_radii,
            min_energies,
            linestyle="None",
            marker="*",
            markeredgewidth=0,
            alpha=alpha,
        )

    def setup_axis(self, ax):
        Lf_scaled = int(self._args["Lf"] / 1e-6)
        Nmin = self._args["Nmin"]
        ax.set_title(
            rf"$N_\text{{sca}} = {Nmin}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$",
            loc="center",
        )
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$\upDelta \Phi / \si{\kb} T$ \hspace{10pt}", labelpad=-2)


class RadiusEnergyNminPlot(Plot):
    def plot_figure(self, f, ax):
        Nmins = self._args["Nmins"]
        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Nmins[0], Nmins[-1])
        energies = []
        radiis = []
        min_energies = []
        min_radii = []
        samples = self._args["samples"]
        N = self._args["N"]
        for Nmin in Nmins:
            max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)
            radii_scaled = radii / 1e-6
            radiis.append(radii_scaled)
            energy = [
                tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
            ]
            energy_scaled = np.array(energy) / (constants.k * self._args["T"])
            energies.append(energy_scaled)
            min_energy_i = np.argmin(energy_scaled)
            min_energies.append(energy_scaled[min_energy_i])
            min_radii.append(np.min(radii_scaled[min_energy_i]))

        # Scale energy to have min at 0
        # energies = np.array(energies)
        # min_energy = np.min(min_energies)
        # energies -= min_energy
        # min_energies -= min_energy

        # Plot
        for Nmin, energy, radii in zip(Nmins, energies, radiis):
            label = rf"$N_\text{{sca}}={Nmin}$"
            ax.plot(radii, energy, color=mappable.to_rgba(Nmin), label=label)

        ax.plot(
            min_radii, min_energies, linestyle="None", marker="*", markeredgewidth=0
        )

    def setup_axis(self, ax):
        Lf_scaled = int(self._args["Lf"] / 1e-6)
        N = self._args["N"]
        ax.set_title(
            rf"$N_\text{{f}} = {N}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$",
            loc="center",
        )
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"\hspace{15pt} $\upDelta \Phi / \si{\kb} T$", labelpad=-4)


class RadiusForceLfPlot(Plot):
    def plot_figure(self, f, ax, calc_degens=False, alpha=1):
        Lfs = self._args["Lfs"]
        N = self._args["N"]
        Nmin = self._args["Nmin"]
        delta = self._args["delta"]
        temp = self._args["T"]

        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Lfs[0], Lfs[-1])
        for i, Lf in enumerate(Lfs):
            self._args["Lf"] = Lf
            if calc_degens:
                lf = self._args["lfs"][i]
                max_height = int(Nmin * lf - Nmin - 1)
                min_height = int((Nmin // 2) * lf - 1)
                heights = range(min_height, max_height + 1)
                heights = np.array(heights)
                radii = (heights + 1) * delta / (2 * np.pi)
                max_radius = radii[-1]
                min_radius = radii[0]
                energies = [
                    tracks_model.calc_ring_energy(r, N, Nmin, self._args) for r in radii
                ]
                energies = np.array(energies)
                degens = tracks_model.calc_degeneracies(
                    heights, lf, N, include_height=False
                )
                boltz_weights = degens * np.exp(-energies / constants.k / temp)
                energies = -constants.k * temp * np.log(boltz_weights)
                force = -np.diff(energies) / (delta / (2 * np.pi))
                radii = radii[1:] - delta
            else:
                samples = self._args["samples"]
                max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
                min_radius = analytical.calc_min_radius(max_radius)
                radii = np.linspace(min_radius, max_radius, num=samples)
                force = [
                    tracks_model.calc_ring_force(r, N, Nmin, self._args) for r in radii
                ]

            # Convert units
            radii_scaled = radii / 1e-6
            force_scaled = np.array(force) / 1e-12

            # Plot
            Lf_scaled = np.round(Lf / 1e-6, decimals=1)
            label = rf"$L_\text{{f}}=\SI{{{Lf_scaled}}}{{\micro\meter}}$"
            ax.plot(
                radii_scaled,
                force_scaled,
                color=mappable.to_rgba(Lf),
                label=label,
                alpha=alpha,
            )

        ax.axhline(0, linestyle="dashed")

    def setup_axis(self, ax):
        Nmin = self._args["Nmin"]
        N = self._args["N"]
        ax.set_title(rf"$N_\text{{sca}} = {Nmin}$, $N_\text{{f}} = {N}$", loc="center")
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$F / \si{\pico\newton}$")
        ax.set_ylim(top=4)


class RadiusForceNPlot(Plot):
    def plot_figure(self, f, ax, calc_degens=False, alpha=1):
        Ns = self._args["Ns"]
        Nmin = self._args["Nmin"]
        delta = self._args["delta"]
        temp = self._args["T"]
        if calc_degens:
            lf = self._args["lf"]
            max_height = Nmin * lf - Nmin - 1
            min_height = (Nmin // 2) * lf - 1
            heights = range(min_height, max_height + 1)
            heights = np.array(heights)
            radii_inp = (heights + 1) * delta / (2 * np.pi)
            radii = radii_inp[1:] - delta
        else:
            samples = self._args["samples"]
            max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)

        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])
        for N in Ns:
            if calc_degens:
                energies = [
                    tracks_model.calc_ring_energy(r, N, Nmin, self._args)
                    for r in radii_inp
                ]
                energies = np.array(energies)
                degens = tracks_model.calc_degeneracies(
                    heights, lf, N, include_height=False
                )
                boltz_weights = degens * np.exp(-energies / constants.k / temp)
                energies = -constants.k * temp * np.log(boltz_weights)
                force = -np.diff(energies) / (delta / (2 * np.pi))
            else:
                force = [
                    tracks_model.calc_ring_force(r, N, Nmin, self._args) for r in radii
                ]

            # Convert units
            radii_scaled = radii / 1e-6
            force_scaled = np.array(force) / 1e-12

            # Plot
            label = rf"$N_\text{{f}}={N}$"
            if calc_degens:
                label = label + ", with degenarcy"
            ax.plot(
                radii_scaled,
                force_scaled,
                color=mappable.to_rgba(N),
                label=label,
                alpha=alpha,
            )

        ax.axhline(0, linestyle="dashed")

    def setup_axis(self, ax):
        Lf_scaled = int(self._args["Lf"] / 1e-6)
        Nmin = self._args["Nmin"]
        ax.set_title(
            rf"$N_\text{{sca}} = {Nmin}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$",
            loc="center",
        )
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$F / \si{\pico\newton}$", labelpad=-1)

    #    ax.set_ylim(bottom=-2)


class RadiusForceNminPlot(Plot):
    def plot_figure(self, f, ax):
        Nmins = self._args["Nmins"]
        N = self._args["N"]
        samples = self._args["samples"]
        cmap = styles.create_truncated_colormap(0.2, 0.8, name="plasma")
        mappable = styles.create_linear_mappable(cmap, Nmins[0], Nmins[-1])

        for Nmin in Nmins:
            max_radius = analytical.calc_max_radius(self._args["Lf"], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)
            force = [
                tracks_model.calc_ring_force(r, N, Nmin, self._args) for r in radii
            ]

            # Convert units
            radii_scaled = radii / 1e-6
            force_scaled = np.array(force) / 1e-12

            # Plot
            label = rf"$N_\text{{sca}}={Nmin}$"
            ax.plot(
                radii_scaled, force_scaled, color=mappable.to_rgba(Nmin), label=label
            )

        ax.axhline(0, linestyle="dashed")

    def setup_axis(self, ax):
        N = self._args["N"]
        Lf_scaled = int(self._args["Lf"] / 1e-6)
        ax.set_title(
            rf"$N_\text{{f}} = {N}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$",
            loc="center",
        )
        ax.set_xlabel(r"$R / \si{\micro\meter}$")
        ax.set_ylabel(r"$F / \si{\pico\newton}$", labelpad=-1)
