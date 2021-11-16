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
    f.savefig(plot_filebase + '.pgf', transparent=True)
    f.savefig(plot_filebase + '.pdf', transparent=True)
    f.savefig(plot_filebase + '.png', transparent=True)


def set_line_labels_by_pos(line, ax, xpos, ha, va, ypos=None, yshift=0):
    xdata = line.get_xdata()
    if ypos is None:
        ypos = line.get_ydata()[np.abs(xdata - xpos).argmin()]
    ax.text(
        xpos, ypos + yshift, line.get_label(), color=line.get_color(),
        horizontalalignment=ha, verticalalignment=va)


def set_line_labels_by_index(line, ax, index, ha, va, xshift=0, yshift=0):
    xpos = line.get_xdata()[index]
    ypos = line.get_ydata()[index]
    ax.text(
        xpos + xshift, ypos + yshift, line.get_label(), color=line.get_color(),
        horizontalalignment=ha, verticalalignment=va)

def set_line_labels_to_middle(line, ax, ha, va, xshift=0, yshift=0):
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    middle_index = (np.argmax(ydata) + np.argmin(ydata))//2
    xpos = xdata[middle_index]
    ypos = ydata[middle_index]
    ax.text(
        xpos, ypos + yshift, line.get_label(), color=line.get_color(),
        horizontalalignment=ha, verticalalignment=va)

def load_sim_params(args, filename):
    with open(filename) as file:
        parms = json.load(file)

    parms['N'] = parms['Nfil']
    parms['Nmin'] = parms['Nsca']
    parms['temp'] = parms['T']

    return args | parms

class Plot:
    def __init__(self, args):
        self._args = args

    def set_labels(self, ax):
        plt.legend()


class FreqsPlot(Plot):
    def plot_figure(self, f, ax):
        itr = self._args['itr']
        input_dir = self._args["input_dir"]
        for vari in self._args['varis']:
            for rep in range(1, self._args['reps'] + 1):
                filename = f'{input_dir}/{vari}/{vari}_rep-{rep}.freqs'
                freqs = pd.read_csv(filename, header=0, delim_whitespace=True)
                heights = freqs.columns.astype(int)

                #ax.plot(heights, np.log10(freqs.iloc[itr - 1]))
                ax.plot(heights, freqs.iloc[itr - 1])

    def setup_axis(self, ax):
        ax.set_ylabel(r'Fraction')
        ax.set_xlabel(r'Lattice height')


class LFEsPlot(Plot):
    def plot_figure(self, f, ax):
        itr = self._args['itr']
        input_dir = self._args['input_dir']
        for vari in self._args['varis']:
            for rep in range(1, self._args['reps'] + 1):
                filename = f'{input_dir}/{vari}/{vari}_rep-{rep}.biases'
                biases = pd.read_csv(filename, header=0, delim_whitespace=True)
                heights = biases.columns.astype(int)
                lfes = -biases / (self._args['temp']*constants.k)

                ax.plot(heights, lfes.iloc[itr - 1], marker='o')

    def setup_axis(self, ax):
        ax.set_ylabel(r'$k_\mathrm{b}T$')
        ax.set_xlabel('Lattice height')


class RadiusLFEsAnalyticalPlot(Plot):
    def plot_figure(self, f, ax, calc_degen=False):
        itr = self._args['itr']
        input_dir = self._args['input_dir']
        vari = self._args['vari']
        delta = self._args['delta']
        Lf = self._args['Lf']
        Nmin = self._args['Nmin']
        N = self._args['N']
        temp = self._args['temp']
        lf = self._args['lf']

        cmap = cm.get_cmap('tab10')

        filename = f'{input_dir}/{vari}/{vari}_rep-1.biases'
        biases = pd.read_csv(filename, header=0, delim_whitespace=True)
        heights = biases.columns.astype(int)
        radii = (heights + 1) * delta / (2*np.pi)
        radii_scaled = radii / 1e-6

        align_i = -1

        for rep in range(1, self._args['reps'] + 1):
            filename = f'{input_dir}/{vari}/{vari}_rep-{rep}.biases'
            biases = pd.read_csv(filename, header=0, delim_whitespace=True)
            heights = biases.columns.astype(int)
            lfes = -biases / (self._args['temp']*constants.k)
            lfes_last_itr = lfes.iloc[-1]
            lfes_last_itr -= lfes_last_itr[align_i]
            ax.plot(radii_scaled, lfes_last_itr, marker='o', color=cmap(0))

        energies = [tracks_model.calc_ring_energy(
            r, N, Nmin, self._args) for r in radii]
        energies = np.array(energies)
        energies_scaled = energies/(constants.k*temp)
        energies_scaled -= energies_scaled[align_i]

        ax.plot(radii_scaled, energies_scaled, color=cmap(1))

        if calc_degen:
            degens = tracks_model.calc_degeneracies(heights, lf)
            boltz_weights = degens*np.exp(-energies/constants.k/temp)
            lfes_anal = -np.log(boltz_weights / boltz_weights[align_i])

            ax.plot(radii_scaled, lfes_anal, color=cmap(2))


    def setup_axis(self, ax):
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'$k_\mathrm{b}T$')


class RadiiPlot(Plot):
    def plot_figure(self, f, ax):
        vari = self._args['vari']
        input_dir = self._args["input_dir"]
        rep = self._args['rep']
        itr = self._args['itr']
        filename = f'{input_dir}/{vari}/{vari}_rep-{rep}_iter-{itr}.ops'
        ops = pd.read_csv(filename, header=0, delim_whitespace=True)
        ax.plot(ops['step'], ops['radius'])

    def setup_axis(self, ax):
        ax.set_ylabel(r'Radius')
        ax.set_xlabel(r'Step')


class LfEradiusNPlot(Plot):
    def plot_figure(self, f, ax):
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        Lfs = self._args['Lfs']
        Ns = self._args['Ns']
        Nmin = self._args['Nmin']
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])
        xvals = []
        yvals = []
        for N in Ns:
            radii = []
            lengths = []
            for j, Lf in enumerate(Lfs):
                self._args['Lf'] = Lf
                max_radius = analytical.calc_max_radius(
                    self._args['Lf'], Nmin)
                min_radius = analytical.calc_min_radius(max_radius)
                r = tracks_model.calc_equilibrium_ring_radius(
                    N, Nmin, self._args)
                if r >= min_radius and r <= max_radius:
                    radii.append(r)
                    lengths.append(Lfs[j])

            # Convert units
            radii = np.array(radii) / 1e-6
            lengths = np.array(lengths) / 1e-6

            # Plot
            label = rf'$N_\text{{f}}={N}'
            ax.plot(lengths, radii, color=mappable.to_rgba(N), label=label)

    def setup_axis(self, ax):
        ax.set_title(
            rf'$N_\text{{sca}} = {self._args["Nmin"]}$',
            loc='center')
        ax.set_xlabel(r'$L^\text{f} / \si{\micro\meter}$')
        ax.set_ylabel(r'$R_\text{eq} / \si{\micro\meter}$')


class LfEradiusNminPlot(Plot):
    def plot_figure(self, f, ax):
        Nmins = self._args['Nmins']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, Nmins[0], Nmins[-1])
        N = self._args['N']
        Lfs = self._args['Lfs']
        for Nmin in Nmins:
            radii = []
            lengths = []
            for j, Lf in enumerate(Lfs):
                self._args['Lf'] = Lf
                max_radius = analytical.calc_max_radius(
                    self._args['Lf'], Nmin)
                min_radius = analytical.calc_min_radius(max_radius)
                r = tracks_model.calc_equilibrium_ring_radius(
                    N, Nmin, self._args)
                if r >= min_radius and r <= max_radius:
                    radii.append(r)
                    lengths.append(Lfs[j])

            # Convert units
            radii = np.array(radii) / 1e-6
            lengths = np.array(lengths) / 1e-6

            # Plot
            label = rf'$N_\text{{sca}}={Nmin}'
            ax.plot(lengths, radii, color=mappable.to_rgba(Nmin), label=label)

    def setup_axis(self, ax):
        ax.set_title(
            rf'$N_\text{{f}} = {self._args["N"]}$',
            loc='center')
        ax.set_xlabel(r'$L^\text{f} / \si{\micro\meter}$')
        ax.set_ylabel(r'$R_\text{eq} / \si{\micro\meter}$')


class XcForcePlot(Plot):
    def plot_figure(self, f, ax):
        fractions = self._args['fractions']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, fractions[0], fractions[-1])
        N = self._args['N']
        Nmin = self._args['Nmin']
        Xcs = self._args['Xcs']
        Lf = self._args['Lf']
        max_radius = analytical.calc_max_radius(Lf, Nmin)
        for fraction in fractions:
            r = max_radius * fraction
            forces = []
            for Xc in Xcs:
                self._args['Xc'] = Xc
                force = tracks_model.calc_ring_force(
                    r, N, Nmin, self._args)
                forces.append(force)

            # Convert units
            forces_scaled = np.array(forces) / 1e-12

            # Plot
            if fraction == 1:
                label = r'$R_{\text{max}}$'
            else:
                label = fr'${fraction} R_{{\text{{max}}}}$'

            ax.plot(Xcs, forces_scaled, color=mappable.to_rgba(fraction),
                    label=label)

        ax.axhline(0, linestyle='dashed')

    def setup_axis(self, ax):
        Lf_scaled = self._args['Lf']/1e-6
        N = self._args['N']
        Nmin = self._args['Nmin']
        ax.set_title(
            rf'$N_\text{{sca}} = {Nmin}$, $N_\text{{f}} = {N}$, $L_\text{{f}}'
            rf'= \SI{{{int(Lf_scaled)}}}{{\micro\meter}}$',
            loc='center')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\text{[X]} / \si{\molar}$')
        ax.set_ylabel(r'$F / \si{\pico\newton}$')
        ax.set_ylim(top=3)
        minor_ticks = ticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9))
        ax.xaxis.set_minor_locator(minor_ticks)
        #ax.set_yscale('log')
        #minor_ticks = ticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9))
        #ax.yaxis.set_minor_locator(minor_ticks)


class RadiusEnergyLfPlot(Plot):
    def plot_figure(self, f, ax):
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        Lfs = self._args['Lfs']
        mappable = styles.create_linear_mappable(cmap, Lfs[0], Lfs[-1])
        energies = []
        radiis = []
        min_energies = []
        min_radii = []
        Nmin = self._args['Nmin']
        N = self._args['N']
        samples = self._args['samples']
        for Lf in Lfs:
            self._args['Lf'] = Lf
            max_radius = analytical.calc_max_radius(self._args['Lf'], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)
            radii_scaled = radii / 1e-6
            radiis.append(radii_scaled)
            energy = [tracks_model.calc_ring_energy(
                r, N, Nmin, self._args) for r in radii]
            energy_scaled = np.array(energy)/(constants.k*self._args['T'])
            energies.append(energy_scaled)
            min_energy_i = np.argmin(energy_scaled)
            min_energies.append(energy_scaled[min_energy_i])
            min_radii.append(np.min(radii_scaled[min_energy_i]))

        # Scale energy to have min at 0
        energies = np.array(energies)
        #min_energy = np.min(min_energies)
        #energies -= min_energy
        #min_energies -= min_energy

        # Plot
        for radii, Lf, energy in zip(radiis, Lfs, energies):
            Lf_scaled = Lf / 1e-6
            label = rf'$L_\text{{f}}=\SI{{{Lf_scaled}}}{{\micro\meter}}$'
            ax.plot(radii, energy, color=mappable.to_rgba(Lf),
                    label=label)

        ax.plot(min_radii, min_energies, linestyle='None',
                marker='*', markeredgewidth=0)

    def setup_axis(self, ax):
        Nmin = self._args['Nmin']
        N = self._args['N']
        ax.set_title(
            rf'$N_\text{{sca}} = {Nmin}$, $N_\text{{f}} = {N}$',
            loc='center')
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'$\upDelta \Phi / \si{\kb} T$ \hspace{7pt}', labelpad=-4)
    #    ax.set_ylim(top=30)
    #    ax.set_xlim(left=3, right=8.7)


class RadiusEnergyNPlot(Plot):
    def plot_figure(self, f, ax):
        Ns = self._args['Ns']
        samples = self._args['samples']
        Nmin = self._args['Nmin']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])
        max_radius = analytical.calc_max_radius(self._args['Lf'], Nmin)
        min_radius = analytical.calc_min_radius(max_radius)
        radii = np.linspace(min_radius, max_radius, num=samples)
        radii_scaled = radii / 1e-6
        energies = []
        min_energies = []
        min_radii = []
        for N in Ns:
            energy = [tracks_model.calc_ring_energy(
                r, N, Nmin, self._args) for r in radii]
            energy_scaled = np.array(energy)/(constants.k*self._args['T'])
            energies.append(energy_scaled)
            min_energy_i = np.argmin(energy_scaled)
            min_energies.append(energy_scaled[min_energy_i])
            min_radii.append(np.min(radii_scaled[min_energy_i]))

        # Scale energy to have min at 0
        energies = np.array(energies)
        #min_energy = np.min(min_energies)
        #energies -= min_energy
        #min_energies -= min_energy

        # Plot
        for N, energy in zip(Ns, energies):
            label = rf'$N_\text{{f}}={N}$'
            ax.plot(radii_scaled, energy, color=mappable.to_rgba(N),
                    label=label)

        ax.plot(min_radii, min_energies, linestyle='None',
                marker='*', markeredgewidth=0)

    def setup_axis(self, ax):
        Lf_scaled = int(self._args['Lf']/1e-6)
        Nmin = self._args['Nmin']
        ax.set_title(
            rf'$N_\text{{sca}} = {Nmin}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$',
            loc='center')
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'$\upDelta \Phi / \si{\kb} T$ \hspace{10pt}', labelpad=-2)


class RadiusEnergyNminPlot(Plot):
    def plot_figure(self, f, ax):
        Nmins = self._args['Nmins']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, Nmins[0], Nmins[-1])
        energies = []
        radiis = []
        min_energies = []
        min_radii = []
        samples = self._args['samples']
        N = self._args['N']
        for Nmin in Nmins:
            max_radius = analytical.calc_max_radius(self._args['Lf'], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)
            radii_scaled = radii / 1e-6
            radiis.append(radii_scaled)
            energy = [tracks_model.calc_ring_energy(
                r, N, Nmin, self._args) for r in radii]
            energy_scaled = np.array(energy)/(constants.k*self._args['T'])
            energies.append(energy_scaled)
            min_energy_i = np.argmin(energy_scaled)
            min_energies.append(energy_scaled[min_energy_i])
            min_radii.append(np.min(radii_scaled[min_energy_i]))

        # Scale energy to have min at 0
        energies = np.array(energies)
        #min_energy = np.min(min_energies)
        #energies -= min_energy
        #min_energies -= min_energy

        # Plot
        for Nmin, energy, radii in zip(Nmins, energies, radiis):
            label = rf'$N_\text{{sca}}={Nmin}$'
            ax.plot(radii, energy, color=mappable.to_rgba(Nmin),
                    label=label)

        ax.plot(min_radii, min_energies, linestyle='None',
                marker='*', markeredgewidth=0)


    def setup_axis(self, ax):
        Lf_scaled = int(self._args['Lf']/1e-6)
        N = self._args['N']
        ax.set_title(
            rf'$N_\text{{f}} = {N}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$',
            loc='center')
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'\hspace{15pt} $\upDelta \Phi / \si{\kb} T$', labelpad=-4)


class RadiusForceLfPlot(Plot):
    def plot_figure(self, f, ax):
        Lfs = self._args['Lfs']
        N = self._args['N']
        Nmin = self._args['Nmin']
        samples = self._args['samples']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, Lfs[0], Lfs[-1])
        for Lf in Lfs:
            self._args['Lf'] = Lf
            max_radius = analytical.calc_max_radius(self._args['Lf'], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)
            force = [tracks_model.calc_ring_force(
                r, N, Nmin, self._args) for r in radii]

            # Convert units
            radii_scaled = radii / 1e-6
            force_scaled = np.array(force) / 1e-12

            # Plot
            Lf_scaled = Lf / 1e-6
            label = rf'$L_\text{{f}}=\SI{{{Lf_scaled}}}{{\micro\meter}}$'
            ax.plot(radii_scaled, force_scaled, color=mappable.to_rgba(Lf),
                    label=label)

        ax.axhline(0, linestyle='dashed')

    def setup_axis(self, ax):
        Nmin = self._args['Nmin']
        N = self._args['N']
        ax.set_title(
            rf'$N_\text{{sca}} = {Nmin}$, $N_\text{{f}} = {N}$',
            loc='center')
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'$F / \si{\pico\newton}$')
        ax.set_ylim(top=4)


class RadiusForceNPlot(Plot):
    def plot_figure(self, f, ax):
        Ns = self._args['Ns']
        Nmin = self._args['Nmin']
        samples = self._args['samples']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, Ns[0], Ns[-1])
        max_radius = analytical.calc_max_radius(self._args['Lf'], Nmin)
        min_radius = analytical.calc_min_radius(max_radius)
        radii = np.linspace(min_radius, max_radius, num=samples)

        for N in Ns:
            force = [tracks_model.calc_ring_force(
                r, N, Nmin, self._args) for r in radii]

            # Convert units
            radii_scaled = radii / 1e-6
            force_scaled = np.array(force) / 1e-12

            # Plot
            label = rf'$N_\text{{f}}={N}$'
            ax.plot(radii_scaled, force_scaled, color=mappable.to_rgba(N),
                    label=label)

        ax.axhline(0, linestyle='dashed')


    def setup_axis(self, ax):
        Lf_scaled = int(self._args['Lf']/1e-6)
        Nmin = self._args['Nmin']
        ax.set_title(
            rf'$N_\text{{sca}} = {Nmin}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$',
            loc='center')
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'$F / \si{\pico\newton}$', labelpad=-1)
    #    ax.set_ylim(bottom=-2)


class RadiusForceNminPlot(Plot):
    def plot_figure(self, f, ax):
        Nmins = self._args['Nmins']
        N = self._args['N']
        samples = self._args['samples']
        cmap = styles.create_truncated_colormap(0.2, 0.8, name='plasma')
        mappable = styles.create_linear_mappable(cmap, Nmins[0], Nmins[-1])

        for Nmin in Nmins:
            max_radius = analytical.calc_max_radius(self._args['Lf'], Nmin)
            min_radius = analytical.calc_min_radius(max_radius)
            radii = np.linspace(min_radius, max_radius, num=samples)
            force = [tracks_model.calc_ring_force(
                r, N, Nmin, self._args) for r in radii]

            # Convert units
            radii_scaled = radii / 1e-6
            force_scaled = np.array(force) / 1e-12

            # Plot
            label = rf'$N_\text{{sca}}={Nmin}$'
            ax.plot(radii_scaled, force_scaled, color=mappable.to_rgba(Nmin),
                    label=label)

        ax.axhline(0, linestyle='dashed')


    def setup_axis(self, ax):
        N = self._args['N']
        Lf_scaled = int(self._args['Lf']/1e-6)
        ax.set_title(
            rf'$N_\text{{f}} = {N}$, $L_\text{{f}} = \SI{{{Lf_scaled}}}{{\micro\meter}}$',
            loc='center')
        ax.set_xlabel(r'$R / \si{\micro\meter}$')
        ax.set_ylabel(r'$F / \si{\pico\newton}$', labelpad=-1)
