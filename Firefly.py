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

        self.ready = False
        self.theta_star = tstar
        self.timestepsize = timestepsize

        self.sign = 1

        # integrate and fire params
        self.beta = beta
        self.charging_time = self.sample_ct()
        self.discharging_time = self.sample_dt()
        self.voltage_threshold = 1
        self.epsilon_delta = epsilon_delta
        self.discharging_threshold = 0.99 #epsilon_delta[1] - 0.01
        self.charging_threshold = 0.0 #epsilon_delta[0]
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
