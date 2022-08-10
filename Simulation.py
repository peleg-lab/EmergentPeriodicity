import math
import pickle
import random

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.signal import find_peaks, peak_prominences

import Firefly
import simulation_helpers

IS_TEST = False


class Simulation:
    def __init__(self, num_agents, side_length, step_count, thetastar, coupling_strength, Tb,
                 beta, phrase_duration, epsilon_delta,timestepsize,r_or_u="uniform", use_linear=False, one_flash=False,
                 no_refrac=False):
        self.firefly_array = []
        self.timestepsize = timestepsize
        self.use_integrate_and_fire = True
        self.beta = beta
        self.phrase_duration = phrase_duration
        self.use_linear = use_linear
        self.one_flash = one_flash
        self.no_refrac = no_refrac

        # constants set by run.py
        self.total_agents = num_agents
        self.n = side_length
        self.coupling_strength = coupling_strength
        self.alpha = 2
        self.Tb = Tb
        self.steps = step_count
        self.r_or_u = r_or_u
        self.tstar_seed = thetastar
        self.epsilon_delta = epsilon_delta
        thetastars = [np.linspace(-thetastar, thetastar, 1)]
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])

        self.has_run = False
        self.obstacles = None

        # initialize all Firefly agents
        for i in range(0, self.total_agents):
            self.firefly_array.append(Firefly.Firefly(
                i, total=self.total_agents, tstar=self.thetastar,
                tstar_range=1,
                n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                beta=beta,
                phrase_duration=phrase_duration,
                epsilon_delta=epsilon_delta,
                use_periodic_boundary_conditions=False,
                timestepsize=0.01,
                use_linear=use_linear,
                one_flash=one_flash,
                no_refrac=no_refrac)
            )
        self.boilerplate = '{}density, {}beta, {}Tb'.format(self.total_agents /
                                                            (self.n * self.n),
                                                            beta, phrase_duration)
        self.init_stats()

    def init_stats(self):
        """Initialize per-timestep dictionaries tracking firefly phase and TODO: more things."""
        initial_flashers = [(ff.positionx[0], ff.positiony[0]) for ff in self.firefly_array if ff.flashed_at_this_step[0]]

    def run(self):
        """
        Run the simulation. 
        """
        logging = False
        for step in range(1, self.steps):
            if logging:
                print(step)
            if step % 5000 == 0:
                print(step)
            if self.use_integrate_and_fire:
                self.lco_interactions(step)

        self.has_run = True

    def look(self, step):
        """Find neighbors in line of sight. Set limited to true to explore varying the FoV.
        Options:
        1. All-to-all adjacency matrix.
        """

        neighbors = {}
        for i in range(0, len(self.firefly_array)):
            ff_i = self.firefly_array[i]
            neighbors[ff_i.number] = []
            for j in range(0, self.total_agents):
                if i == j:
                    # same agent
                    continue
                else:
                    ff_j = self.firefly_array[j]
                    if ff_j.flashed_at_this_step[step - 1]:
                        neighbors[ff_i.number].append(ff_j)

        return neighbors

    def listen(self, step, neighbors):
        """Update voltages based on adjacency matrix"""
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]
            dvt = ff_i.set_dvt(step, ff_i.in_burst)
            neighbors_of_i = neighbors[ff_i.number]
            beta_addition = 0
            if neighbors_of_i:
                for ff_j in neighbors_of_i:
                    beta_addition += (ff_i.beta * (1 - ff_j.is_charging))

            voltage_at_step = ff_i.voltage_instantaneous[step - 1] + (dvt + (ff_i.sign * beta_addition))
            ff_i.voltage_instantaneous[step] = min([ff_i.discharging_threshold, voltage_at_step])

    def lco_interactions(self, step):
        """
        At timestep step:
        A) The neighbor set is chosen in _look_ (here it is all-to-all)
        B) Agents receive pulsatile inputs from their neighbors in _listen_
        C) Agents update their voltages and epsilon parameters in _update_epsilon_and_readiness_.
        """
        neighbors = self.look(step)
        self.listen(step, neighbors)
        self.update_epsilon_and_readiness(step)

    def update_epsilon_and_readiness(self, step):
        """Set epsilon based on voltage for all fireflies, flash if possible."""
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]

            # update epsilon to discharging (V is high enough)
            if ff_i.voltage_instantaneous[step] >= ff_i.discharging_threshold:
                if ff_i.in_burst is False and ff_i.sign == 1:
                    if ff_i.flashes_per_burst == 1 and step - ff_i.last_flashed_at > (ff_i.discharging_time / ff_i.timestepsize):
                        # on its own
                        ff_i.set_ready(step)
                    elif ff_i.flashes_per_burst > 1:
                        ff_i.set_ready(step)
                elif ff_i.in_burst is True and ff_i.sign == 1:
                    # in a burst
                    ff_i.unset_ready()
                else:
                    ff_i.unset_ready()
                if ff_i.sign == 1:
                    ff_i.is_charging = 0
                    ff_i.epsilon[step] = 0

            # update epsilon to charging if agent flashes
            elif ff_i.voltage_instantaneous[step] <= ff_i.charging_threshold:
                if ff_i.in_burst is False:
                    ff_i.is_charging = 1
                    ff_i.epsilon[step] = 1
                ff_i.unset_ready()
            self.flash_conditions(ff_i, step)

    @staticmethod
    def flash_conditions(ff, t):
        """When to flash for firefly ff at step t"""
        if ff.is_charging == 0 and ff.ready and not ff.in_burst:
            ff.flash(t)

        elif ff.in_burst is True and t - ff.last_flashed_at > (ff.discharging_time + ff.charging_time):
            ff.flash(t)

    def set_save_string(self, plot_type, now, path):
        """Sets up default save string."""
        if 'phaseanim' in plot_type or 'numphaseovertime' in plot_type:
            end = '.mp4'
        elif 'burst_dict' in plot_type:
            end = '.pickle'
        else:
            end = '.png'
        if not path:
            save_string = 'simulation_results/{}_{}agents_{}x{}_beta={}_Tb={}_k={}_steps={}_{}distribution{}_obstacles{}'.format(
                plot_type,
                self.total_agents,
                self.n, self.n,
                self.beta,
                self.phrase_duration,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                str(now).replace(' ', '_'),
                end
            )
        else:
            save_string = '{}{}_{}agents_{}x{}_beta={}_Tb={}_k={}_steps={}_{}distribution{}{}'.format(
                path,
                plot_type,
                self.total_agents,
                self.n, self.n,
                self.beta,
                self.phrase_duration,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                str(now).replace(' ', '_'),
                end
        )
        return save_string

    def calc_interburst_distribution(self):
        """Calculate the distribution of interburst intervals for all individuals in a simulation.
        :returns: Flat list of interburst distributions
        """
        starts_of_bursts = {}
        for firefly in self.firefly_array:
            starts_of_bursts[firefly.number] = []
            flashes = firefly.flashes_per_burst
            for i, yes in enumerate(firefly.flashed_at_this_step):
                if yes and flashes == firefly.flashes_per_burst:
                    starts_of_bursts[firefly.number].append(i)
                    flashes -= 1
                else:
                    if yes:
                        flashes -= 1
                        if flashes == 0:
                            flashes = firefly.flashes_per_burst

        interburst_distribution = [[starts_of_bursts[a][i+1] - starts_of_bursts[a][i]
                                   for i in range(len(starts_of_bursts[a])-1)]
                                   for a in starts_of_bursts.keys()]
        flat_interburst_distribution = [item for sublist in interburst_distribution for item in sublist]

        return flat_interburst_distribution

    def temporal_interburst_dist(self):
        """Returns dict of inter-burst intervals over time."""
        starts_of_bursts = {}
        for firefly in self.firefly_array:
            j = 0
            starts_of_bursts[firefly.number] = []
            flashes = firefly._flashes_per_burst[j]
            for i, yes in enumerate(firefly.flashed_at_this_step[0:]):
                if yes and flashes == firefly._flashes_per_burst[j]:
                    starts_of_bursts[firefly.number].append(i)
                    flashes -= 1
                else:
                    if yes:
                        flashes -= 1
                        if flashes == 0:
                            j += 1
                            flashes = firefly._flashes_per_burst[j]
        longest_list = max(list(starts_of_bursts.values()), key=lambda l: len(l))
        number_of_bursts = len(longest_list)

        # pad shorties
        for k, burst in starts_of_bursts.items():
            if len(burst) < number_of_bursts:
                starts_of_bursts[k].extend([float("inf")] * (number_of_bursts - len(burst)))

        collective_burst_starts = []
        for index in range(0, number_of_bursts):
            starting_points = np.array([burst[index] for burst in list(starts_of_bursts.values())])
            collective_burst_starts.append(np.mean(starting_points[starting_points < 1000000]))

        temporal_interbursts = {}
        for i in range(len(collective_burst_starts) - 1):
            interburst = collective_burst_starts[i + 1] - collective_burst_starts[i]
            temporal_interbursts[i] = interburst
        return temporal_interbursts

    def swarm_interburst_dist(self,  is_one=True, is_null=False):
        """Calculate the distribution of interburst intervals for the collective bursting events.
        :returns: Flat list of interburst distributions
        """
        if is_one:
            all_flashes = []
            for firefly in self.firefly_array:
                all_flashes.extend([x for x in firefly.starts_of_bursts])
            all_flashes = sorted(all_flashes)
            isis = [max(abs(j - i), abs(k - j)) for i, j, k in zip(all_flashes, all_flashes[1:], all_flashes[2:])]
            return np.array(isis)
        else:
            starts_of_bursts = {}
            for firefly in self.firefly_array:
                starts_of_bursts[firefly.number] = firefly.starts_of_bursts

            longest_list = max(list(starts_of_bursts.values()), key=lambda l: len(l))
            number_of_bursts = len(longest_list)

            for k, burst in starts_of_bursts.items():
                if len(burst) < number_of_bursts:
                    starts_of_bursts[k].extend([float("inf")] * (number_of_bursts - len(burst)))

            collective_burst_starts = []
            for index in range(0, number_of_bursts):
                starting_points = np.array([burst[index] for burst in list(starts_of_bursts.values())])
                collective_burst_starts.append(np.mean(starting_points[starting_points < 1000000]))
            collective_interburst_distribution = np.array([collective_burst_starts[i+1] - collective_burst_starts[i]
                                                           for i in range(len(collective_burst_starts)-1)])
            temporal_interbursts = {}
            for i in range(len(collective_interburst_distribution) - 1):
                interburst = collective_interburst_distribution[i + 1] - collective_interburst_distribution[i]
                temporal_interbursts[i] = interburst

            peaks, _, last_high_step, _ = self.peak_variances(thresh=0)

            _collective_interburst_distribution = [peaks[i+1] - peaks[i]
                                                   for i in range(len(peaks)-1)
                                                   ]
            if not is_null:
                _collective_interburst_distribution = collective_interburst_distribution[
                    collective_interburst_distribution > 0
                ]

            cid = np.array(_collective_interburst_distribution)

            return cid

    def get_burst_data(self):
        """Male bursts.
        :returns dict of flash counts at timesteps
        """
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                if firefly.flashed_at_this_step[step] is True:
                    to_plot[step] += 1
        return to_plot

    def _get_burst_data(self):
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                x = firefly._get_flashed_at_this_step()
                if x[step] is True:
                    to_plot[step] += 1
        return to_plot

