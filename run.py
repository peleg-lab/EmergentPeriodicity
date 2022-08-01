import math
import networkx as nx
import os
import sys
import pickle
import numpy as np
from datetime import datetime
import pathos.multiprocessing as multip
import argparse

import simulation_helpers
import Simulation


TSTARS = "theta*"
TBS = "Tb"
NS = "n"
NUM_AGENTS = "num_agents"
STEPS = "steps"
KS = "k"
TRIALS = "trials"
E_DELTAS = "epsilon_deltas"
BETAS = "betas"
DBS = "databases"
PHRASE_DURATIONS = "phrases"
TIMESTEPSIZE = "time_step_size"

TRACE_KEY = 'all_paths'
FLASH_KEY = 'all_flash_steps'
BURST_KEY = 'flashes_per_burst'
OBSTACLE_KEY = 'obstacles'
DISTANCE_KEY = 'distances'
STARTS_KEY = 'starts'

DUMP_DATA = True
DUMP_PICKLES = True


def main():
    now = datetime.now()
    use_processes = True

    # can pass 1 or more db files without specifying any other arguments
    if len(sys.argv) > 1 and "-n" not in sys.argv:
        experiment_results = load_pickles(sys.argv)

    # can also pass multiple arguments to run new simulations (agent count, side len, simulation len, trials)
    elif len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num", "-n", type=int, nargs='+', required=True)
        parser.add_argument("--steps", "-s", type=int, required=True)
        parser.add_argument("--length", "-l", type=int, required=True)
        parser.add_argument("--trials", "-t", type=int, required=True)
        parser.add_argument("--beta_range", "-b", type=float, nargs='+', required=False)
        parser.add_argument("--epsilon_lower", "-el", type=float, required=False)
        parser.add_argument("--epsilon_upper", "-eu", type=float, required=False)
        parser.add_argument('--obstacles', dest='obstacles', action='store_true')
        parser.add_argument('--use_linear', dest='use_linear', action='store_true')
        parser.add_argument('--one_flash', dest='one_flash', action='store_true')
        parser.add_argument('--no_refrac', dest='no_refrac', action='store_true')
        parser.add_argument('--time_step_size', type=float, required=False)
        parser.add_argument('--folder', type=str, required=False)
        parser.set_defaults(folder='simulation_results')
        parser.set_defaults(epsilon_lower=0.0)
        parser.set_defaults(epsilon_upper=1)
        parser.set_defaults(time_step_size=0.01)
        parser.set_defaults(no_refrac=False)
        parser.set_defaults(obstacles=False)
        parser.set_defaults(use_linear=False)
        parser.set_defaults(one_flash=False)
        args = parser.parse_args()
        beta_range = extract_range(args.beta_range)
        epsilon_deltas = (args.epsilon_lower, args.epsilon_upper)
        num_list = [int(num) for num in args.num]
        params = set_constants(nao=num_list, sc=args.steps, sl=args.length, nt=args.trials, betas=beta_range,
                               epsilon_deltas=epsilon_deltas, ts=args.time_step_size)

        simulations = setup_simulations(params,
                                        use_obstacles=args.obstacles,
                                        one_flash=args.one_flash,
                                        use_linear=args.use_linear,
                                        no_refrac=args.no_refrac)
        experiment_results = run_simulations(simulations, use_processes=use_processes)
        if DUMP_DATA:
            pickle_results(experiment_results, now, folder=args.folder)

    # or run the default settings
    else:
        params = set_constants()
        simulations = setup_simulations(params)
        experiment_results = run_simulations(simulations, use_processes=use_processes)
        if DUMP_DATA:
            pickle_results(experiment_results, now, folder=folder)
    print("done")


def extract_range(arglist):
    """Process argument list. Arglist can either be empty, one value, or two values constituting a min-max range."""
    if arglist is not None:
        float_args = [float(a) for a in arglist]
        if len(float_args) == 1:
            retlist = float_args
        else:
            retlist = np.arange(float_args[0], float_args[1], 0.05)
    else:
        retlist = None
    return retlist


def pickle_results(experiment_results, now, folder):
    """Dump results to pickle format."""
    for k, v in experiment_results.items():
        try:
            beta, sidelength, number, distribution = k
            density = float(number) / (float(sidelength)*float(sidelength))
            name = '{}density{}beta{}Tb{}_steps'.format(density, beta, distribution, v[0].steps)
        except ValueError:
            beta, sidelength, number, distribution = k
            name = '0.078125density{}betadistributionTb200000_steps'.format(beta)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if not os.path.isdir('{}/{}ff'.format(folder, number)):
            os.makedirs('{}/{}ff'.format(folder, number))
        with open("{}/{}ff/{}_experiment_results_{}.pickle".format(
                folder, number, name, str(now).replace(' ', '_')), 'wb') as f:
            pickle.dump(v, f)


def load_pickles(program_argv):
    """Process pickled data from folder or file

    :param program_argv: argv
    :param experiment_results: dict to update
    """
    experiment_results = {}
    for db in program_argv[1:]:
        if os.path.isdir(os.path.abspath(db)):
            for dbf in os.listdir(os.path.abspath(db)):
                with open(os.path.abspath(db + '/' + dbf), 'rb') as f:
                    try:
                        update = {dbf: pickle.load(f)}
                        experiment_results.update(update)
                    except TypeError:
                        print('Pickle data expected!')
        else:
            with open(os.path.abspath(db), 'rb') as f:
                try:
                    update = {db: pickle.load(f)}
                    experiment_results.update(update)
                except TypeError:
                    print('Pickle data expected!')
    return experiment_results


def set_constants(sl=None, sc=None, nao=None, nt=None, betas=None, epsilon_deltas=None, ts=None):
    """Set up experiment constants.

    :param sl: side length
    :param sc: steps
    :param nao: list of number of agent values
    :param nt: number of trials
    :param betas: list of beta values
    :param epsilon_deltas: list of voltage spans
    :param ts: time step size
    :returns dict of values
    """
    if not ts:
        timestepsize = 0.01
    else:
        timestepsize = ts
    if not sl:
        side_length = 16
    else:
        side_length = sl
    if not nao:
        num_agent_options = [20]
    else:
        num_agent_options = nao
    if not sc:
        step_count = 200000
    else:
        step_count = sc
    if not nt:
        num_trials = 5
    else:
        num_trials = nt
    if betas is None:
        btas = [0.2]
    else:
        btas = betas
    
    epdeltas = [epsilon_deltas]
    params = {}
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    coupling_strengths = [0.03]  # , 0.2, 0.5]
    params[E_DELTAS] = epdeltas
    params[PHRASE_DURATIONS] = ["distribution"]
    params[BETAS] = btas
    params[TSTARS] = thetastars
    params[TBS] = inter_burst_intervals
    params[NS] = side_length
    params[NUM_AGENTS] = num_agent_options
    params[STEPS] = step_count
    params[KS] = coupling_strengths
    params[TRIALS] = num_trials
    params[E_DELTAS] = epdeltas
    params[TIMESTEPSIZE] = timestepsize
    return params


def setup_simulations(params, use_obstacles=False, one_flash=False, use_linear=False, no_refrac=False):
    """
    Instantiate t*n*cs*tb*trial simulation objects with their parameters, where

    t=number of different thetastar ranges,
    n=number of different agent counts,
    cs=number of different coupling strengths,
    tb=number of different internal frequencies,
    trial=number of trials. All these values are held in the params dict.
    Right now, side length and step count are held as constants, but the params dict could easily pass those as lists
    and add to the combinatorics by iterating through each of those as well.
    """
    simulations = generate_simulations(params, use_obstacles, use_linear, one_flash, no_refrac)
    return simulations


def generate_simulations(params, use_obstacles, use_linear, one_flash, no_refrac):
    """Instantiate simulations."""
    for thetastar in params[TSTARS]:
        for num_agents in params[NUM_AGENTS]:
            for coupling_strength in params[KS]:
                for Tb in params[TBS]:
                    for epsilon_delta in params[E_DELTAS]:
                        for beta in params[BETAS]:
                            for phrase_duration in params[PHRASE_DURATIONS]:
                                for trial in range(0, params[TRIALS]):
                                    n = params[NS]
                                    step_count = params[STEPS]
                                    timestepsize = params[TIMESTEPSIZE]
                                    simulation = Simulation.Simulation(num_agents=num_agents,
                                                                       side_length=n,
                                                                       step_count=step_count,
                                                                       thetastar=thetastar,
                                                                       coupling_strength=coupling_strength,
                                                                       Tb=Tb,
                                                                       beta=beta,
                                                                       phrase_duration=phrase_duration,
                                                                       epsilon_delta=epsilon_delta,
                                                                       r_or_u="random",
                                                                       one_flash=one_flash,
                                                                       use_linear=use_linear,
                                                                       no_refrac=no_refrac,
                                                                       timestepsize=timestepsize)
                                    yield simulation


def run_simulations(simulations, use_processes=False):
    """
    Run all simulations set up by setup_simulations.
    The results are stored in a dictionary keyed by their parameters.
    """
    experiment_results = {}
    if use_processes:
        process_pool = multip.ProcessingPool(multip.cpu_count())
        process_results = process_pool.map(run_simulation_in_process, simulations)

        for finished_simulation in process_results:
            result_key = 'beta={}_agents={}'.format(finished_simulation.beta, finished_simulation.total_agents)
            if experiment_results.get(result_key):
                experiment_results[result_key].append(finished_simulation)
            else:
                experiment_results[result_key] = [finished_simulation]

    else:
        for simulation in simulations:
            result_key = 'beta={}_agents={}'.format(simulation.beta, simulation.total_agents)
            if experiment_results.get(result_key):
                experiment_results[result_key].append(simulation)
            else:
                experiment_results[result_key] = [simulation]

    return experiment_results


def run_simulation_in_process(simulation):
    """Wrapper around the multiprocessing run."""
    print('running: with {} agents'.format(simulation.total_agents))
    simulation.run()
    return simulation


if __name__ == "__main__":
    main()
