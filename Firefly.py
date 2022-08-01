import math
import random

import numpy as np

import simulation_helpers


class Firefly:
    def __init__(self, i, total, tstar, tstar_range, n, steps, r_or_u, beta, phrase_duration, epsilon_delta,
                 timestepsize, use_periodic_boundary_conditions=True, use_linear=False, one_flash=False,
                 no_refrac=False, obstacles=None):
        self.steps = steps
        self.simple = False
        if one_flash:
            self.simple = True
        self.numerator = math.log(2)
        if use_linear:
            self.numerator = 1
        self.no_refrac = no_refrac

        self.name = "FF #{}".format(i)
        self.number = i
        if use_periodic_boundary_conditions:
            self.boundary_conditions = self.periodic_boundary_conditions
        else:
            self.boundary_conditions = self.non_periodic_boundary_conditions
        self.velocity = 1.0
        self.side_length_of_enclosure = n
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        if obstacles:
            r_or_u = "random"
        if r_or_u == "random":
            points_set = False
            if obstacles:
                while not points_set:
                    success = True
                    self.positionx[0] = random.randint(0, n)
                    self.positiony[0] = random.randint(0, n)
                    for obstacle in obstacles:
                        if obstacle.contains(self.positionx[0], self.positiony[0]):
                            success = False
                    if success:
                        points_set = True
            else:
                self.positionx[0] = random.randint(0, n)
                self.positiony[0] = random.randint(0, n)
        else:
            uniform_x_position, uniform_y_position = simulation_helpers.get_uniform_coordinates(i, n, total)
            self.positionx[0] = uniform_x_position
            self.positiony[0] = uniform_y_position

        self.direction = np.zeros(steps)
        self.direction[0] = simulation_helpers.get_initial_direction(tstar_range)
        self.direction_set = False
        self.ready = False
        self.theta_star = tstar
        self.timestepsize = timestepsize

        self.sign = 1

        self.phase = np.zeros(steps)
        self.phase[0] = random.random() * math.pi * 2

        # integrate and fire params
        self.beta = beta
        self.charging_time = self.sample_ct()
        self.discharging_time = self.sample_dt()
        self.voltage_threshold = 1
        self.epsilon_delta = epsilon_delta
        self.discharging_threshold = epsilon_delta[1] - 0.01
        self.charging_threshold = epsilon_delta[0]
        self.in_burst = False
        self.voltage_instantaneous = np.zeros(steps)
        self.epsilon = np.zeros(steps)
        self.voltage_instantaneous[0] = random.uniform(self.charging_threshold, self.discharging_threshold)

        if self.voltage_instantaneous[0] > self.discharging_threshold:
            self.is_charging = 0
            self.epsilon[0] = 0
        else:
            self.is_charging = 1
            self.epsilon[0] = 1
        if self.sign == 1:
            self.flashes_per_burst = self.sample_nf()
        else:
            self.flashes_per_burst = 1
            self.charging_time = 60
            self.discharging_time = 1

        self.min_refractory_period = 5.6723333333333334 / self.timestepsize
        self.refractory_period = self.get_refractory_period()
        if phrase_duration == "distribution":
            if self.sign == 1:
                self.phrase_duration = simulation_helpers.draw_from_input_distribution(1) / self.timestepsize
            else:
                # female defaults
                self.phrase_duration = 100
        else:
            self.phrase_duration = phrase_duration  # timesteps, where each timestep = 0.1s

        self.flashes_left_in_current_burst = self.flashes_per_burst
        self._flashes_per_burst = [self.flashes_per_burst]
        self.last_flashed_at = 0
        self.quiet_period = self.phrase_duration - self.refractory_period

        self.flashed_at_this_step = [False] * steps
        self.steps_with_flash = set()
        self.ends_of_bursts = [0]
        self.starts_of_bursts = []

        # the total path of a firefly through 2d space
        self.trace = {0: (self.positionx[0], self.positiony[0])}

        self.nat_frequency = 1 / self.phrase_duration

    def set_ready(self, step):
        if step - self.last_flashed_at > self.refractory_period:
            self.ready = True

    def unset_ready(self):
        self.ready = False

    def get_refractory_period(self):
        if self.no_refrac:
            return 0
        else:
            return self.min_refractory_period / self.timestepsize

    def sample_nf(self):
        if self.simple:
            return 1
        else:
            return int(np.random.choice([2, 3, 4, 5, 6], p=[(1/18), (2/18), (11/18), (3/18), (1/18)]))

    def sample_dt(self):
        # returns a value already in timesteps
        ps = [(1 / 76), (3 / 76), (6 / 76), (4 / 76), (19 / 76), (13 / 76), (14 / 76),
              (11 / 76), (2 / 76), (1 / 76), (1 / 76), 0, 0, 0, 0, 0, 0, 0, (1 / 76)]
        dt = np.random.choice(np.arange(0.1, 2, 0.1), p=ps) / 6  # values in seconds
        dt = dt / self.timestepsize
        return dt

    def sample_ct(self):
        # returns a value already in timesteps
        ps = [(6 / 43), (10 / 43), (12 / 43), (7 / 43), (2 / 43), (2 / 43), (1 / 43), (0 / 43), (2 / 43), (1 / 43)]
        ct = np.random.choice(np.arange(2.6, 3.6, 0.1), p=ps) / 6  # values in seconds
        ct = ct / self.timestepsize
        return ct

    def get_phrase_duration(self):
        return self.phrase_duration

    def _get_flashed_at_this_step(self):
        fats = [False] * self.steps
        for i, t in enumerate(self.flashed_at_this_step):
            if t:
                fats[i] = True
                try:
                    fats[i + 1] = True
                except IndexError:
                    continue
        return fats

    def update_phrase_duration(self, fastest=None):
        if fastest is None:
            self.phrase_duration = simulation_helpers.draw_from_input_distribution(1)  # value in seconds
            self.phrase_duration = self.phrase_duration / self.timestepsize  # value in timesteps
            self.update_quiet_period()
        else:
            self.quiet_period = fastest

    def update_quiet_period(self):
        self.quiet_period = self.phrase_duration - (
            ((self.charging_time + self.discharging_time) * self.flashes_per_burst))

    def move(self, current_step, obstacles, flip_direction=False):
        """Move a firefly through 2d space using a correlated 2d random walk."""
        random_int = random.randint(0, 99)
        step_theta = self.theta_star[random_int]
        decrease_velocity = False
        if current_step == 0:
            direction = self.direction[current_step]
        elif self.direction_set:
            direction = self.direction[current_step - 1]
            self.direction_set = False
        elif flip_direction:
            direction = self.direction[current_step - 1] * -1
            direction = direction + step_theta
            decrease_velocity = True
        else:
            direction = self.direction[current_step - 1] + step_theta

        self.attempt_step(current_step, direction, obstacles, decrease_velocity=decrease_velocity)

    def attempt_step(self, current_step, direction, obstacles, decrease_velocity=False):
        """Stage a step for completion."""
        if decrease_velocity:
            self.velocity /= 2
        potential_x_position = self.positionx[current_step - 1] + self.velocity * math.cos(direction)
        potential_y_position = self.positiony[current_step - 1] + self.velocity * math.sin(direction)

        self.direction[current_step] = direction
        self.complete_step(current_step, potential_x_position, potential_y_position, obstacles)

    def complete_step(self, current_step, x, y, obstacles):
        """Complete a step if it does not interfere with an obstacle; recall move otherwise."""
        self.positionx[current_step] = x
        self.positiony[current_step] = y
        self.boundary_conditions(current_step)
        if obstacles:
            for obstacle in obstacles:
                if obstacle.contains(self.positionx[current_step], self.positiony[current_step]):
                    self.positionx[current_step] = self.positionx[current_step-1]
                    self.positiony[current_step] = self.positiony[current_step-1]
                    self.direction[current_step] = self.direction[current_step-1]
                    self.move(current_step, obstacles, flip_direction=True)
        self.velocity = 1.0
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])

    def periodic_boundary_conditions(self, current_step):
        """Going off the edge of the arena returns an agent to the other side."""
        if self.positionx[current_step] > self.side_length_of_enclosure:
            self.positionx[current_step] = self.positionx[current_step] - self.side_length_of_enclosure
        if self.positionx[current_step] < 0:
            self.positionx[current_step] += self.side_length_of_enclosure

        if self.positiony[current_step] > self.side_length_of_enclosure:
            self.positiony[current_step] = self.positiony[current_step] - self.side_length_of_enclosure
        if self.positiony[current_step] < 0:
            self.positiony[current_step] += self.side_length_of_enclosure

    def non_periodic_boundary_conditions(self, current_step):
        """Bounce off the edges of the arena."""
        flip_direction = False
        if self.positionx[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positionx[current_step] - self.side_length_of_enclosure)
            self.positionx[current_step] = self.positionx[current_step] - 2 * distance_from_edge
            flip_direction = True
        if self.positionx[current_step] < 0:
            distance_from_edge = abs(0 - self.positionx[current_step])
            self.positionx[current_step] = self.positionx[current_step] + 2 * distance_from_edge
            flip_direction = True
        if self.positiony[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positiony[current_step] - self.side_length_of_enclosure)
            self.positiony[current_step] = self.positiony[current_step] - 2 * distance_from_edge
            self.direction[current_step] = -self.direction[current_step]
            flip_direction = True
        if self.positiony[current_step] < 0:
            distance_from_edge = abs(0 - self.positiony[current_step])
            self.positiony[current_step] = self.positiony[current_step] + 2 * distance_from_edge
            flip_direction = True

        if flip_direction:
            self.direction[current_step] = -self.direction[current_step]

    def flash(self, t):
        self.last_flashed_at = t
        self.flashed_at_this_step[t] = True
        self.steps_with_flash.add(t)
        if self.flashes_left_in_current_burst == self.flashes_per_burst:
            self.starts_of_bursts.append(t)
        self.flashes_left_in_current_burst -= 1
        self.in_burst = True
        if self.flashes_left_in_current_burst == 0:

            self.refractory_period = self.get_refractory_period()
            self.flashes_per_burst = self.sample_nf()
            self._flashes_per_burst.append(self.flashes_per_burst)
            self.charging_time = self.sample_ct()
            self.discharging_time = self.sample_dt()
            self.update_phrase_duration()

            self.in_burst = False
            self.unset_ready()
            self.flashes_left_in_current_burst = self.flashes_per_burst
            self.ends_of_bursts.append(t)

    def set_dvt(self, t, in_burst=False):
        prev_voltage = self.voltage_instantaneous[t - 1]
        if not in_burst:
            tc = self.quiet_period
            td = self.discharging_time
        else:
            tc = self.charging_time
            td = self.discharging_time
        if self.is_charging:
            if self.numerator == 1:
                dvt = (self.numerator / tc)
            else:
                dvt = ((self.numerator / tc) * (self.voltage_threshold - prev_voltage))
        else:
            if self.numerator == 1:
                dvt = (-self.numerator / td)
            else:
                dvt = (-(self.numerator / td) * prev_voltage)
        return dvt
