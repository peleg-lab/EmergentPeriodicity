import csv
import math
import random

import pickle

import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas as pd
from joypy import joyplot
from matplotlib import cm

import numpy, scipy.io
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, argrelextrema, argrelmax



def get_uniform_coordinates(i, side_length, total):
    """Distribute a firefly at index i within a uniform distribution."""
    positionsx = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    positionsy = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    x, y = numpy.meshgrid(positionsx, positionsy)
    x_coords = x.flatten()
    y_coords = y.flatten()
    return x_coords[i], y_coords[i]


def test_initial_coordinates():
    """Plot uniform distribution for visual confirmation."""
    total = 150
    side_length = 15
    for i in range(0, total):
        plt.scatter(get_uniform_coordinates(i, side_length, total)[0],
                    get_uniform_coordinates(i, side_length, total)[1])
    plt.show()


def get_initial_direction(theta_star_range):
    """Set up a direction from within a range of angles."""
    all_directions = numpy.linspace(-math.pi, math.pi, theta_star_range)
    return all_directions[random.randint(0, theta_star_range - 1)]


def generate_line_points(pointa, pointb, num_points):
    """"
    Return a list of nb_points equally spaced points
    between p1 and p2
    """
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (pointb[0] - pointa[0]) / (num_points + 1)
    y_spacing = (pointb[1] - pointa[1]) / (num_points + 1)

    return [[pointa[0] + i * x_spacing, pointa[1] + i * y_spacing] for i in range(1, num_points + 1)]


def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    if _len > 0:
        centroid_x = sum(x_coords) / _len
        centroid_y = sum(y_coords) / _len
        return [centroid_x, centroid_y]
    else:
        return None


def get_initial_interburst_interval():
    from scipy.interpolate import make_interp_spline
    import numpy
    import csv

    with open('data/ibs01ff.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    interflash_cutoff = 1.0
    good_data = [float(d[0]) for d in data]
    trimmed_data = [d for d in good_data if d > interflash_cutoff]
    limit = min(trimmed_data)
    n, x = numpy.histogram(trimmed_data, bins=50, density=True)
    bin_centers = 0.5 * (x[1:] + x[:-1])
    ys = [height for height in n]

    # 0s between 0-Tb_min
    pre_x = numpy.arange(0.0, limit, 0.1)
    pre_y = [0.0] * len(pre_x)

    # x_range
    x_nice = numpy.arange(limit, max(bin_centers), 0.1)
    # generate H' func
    _nice = make_interp_spline(bin_centers, ys)
    # calculate H'(x)
    y_nice = _nice(x_nice)
    y_nice = [y if y > 0 else 0 for y in y_nice]

    # H''(x): 0 between 0 and Tb_min, H'(x) between Tb_min and Tb_max
    finalx = numpy.concatenate((pre_x,x_nice))
    finaly = numpy.concatenate((pre_y,y_nice))

    # normalize
    finaly = [float(i)/sum(finaly) for i in finaly]

    # write
    with open('data/envelope_01ff.csv', 'w') as fwrite:
        writer = csv.writer(fwrite)
        for x, y in zip(finalx, finaly):
            writer.writerow([x,y])


def draw_from_input_distribution(num_draws):
    with open('experimental_data/experimental/envelope_01ff.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    X_NICE = [float(d[0]) for d in data]
    Y_NICE = [float(d[1]) for d in data]
    Tc_input = numpy.random.choice(X_NICE, num_draws, p=Y_NICE)
    return Tc_input


def calc_means_stds(interburst_interval_distribution, swarm_interburst_interval_distribution, on_betas=False):
        individual_dicts = [vals for vals in interburst_interval_distribution.values()]
        i_d = {list(individual_dicts[i].keys())[0]: list(individual_dicts[i].values())
               for i in range(len(individual_dicts))}
        keys = i_d.keys()
        individual_means = {k: 0 for k in keys}
        individual_stds = {k: 0 for k in keys}
        for key in keys:
            lvals = [value for list_of_vals in i_d[key] for vals in list_of_vals for value in vals]
            if len(lvals) > 0:
                individual_means[key] = numpy.mean(lvals)
                individual_stds[key] = numpy.std(lvals)
            else:
                individual_means[key] = 'No distribution found'
                individual_stds[key] = 'No distribution found'
        swarm_dicts = [v for v in swarm_interburst_interval_distribution.values()]
        s_d = {list(swarm_dicts[i].keys())[0]: list(swarm_dicts[i].values())
               for i in range(len(swarm_dicts))}
        keys = s_d.keys()
        swarm_means = {k: 0 for k in keys}
        swarm_stds = {k: 0 for k in keys}
        for key in keys:
            lvals = [s_value for s_list_of_vals in s_d[key] for s_vals in s_list_of_vals for s_value in s_vals]
            if len(lvals) > 0:
                swarm_means[key] = numpy.mean(lvals)
                swarm_stds[key] = numpy.std(lvals)
            else:
                swarm_means[key] = 'No distribution found'
                swarm_stds[key] = 'No distribution found'

        return swarm_means, swarm_stds, individual_means, individual_stds


def plots(d, name, measurement_type):
    with open('5ff_peaks_low_density.pickle', 'rb') as f:
        five_ff_peaks_low = pickle.load(f)[::2]
    with open('5ff_peaks_high_density.pickle', 'rb') as f:
        five_ff_peaks_high = pickle.load(f)[::2]

    with open('10ff_peaks_low_density.pickle', 'rb') as f:
        ten_ff_peaks_low = pickle.load(f)[::2]
    with open('10ff_peaks_high_density.pickle', 'rb') as f:
        ten_ff_peaks_high = pickle.load(f)[::2]

    with open('15ff_peaks_low_density.pickle', 'rb') as f:
        fifteen_ff_peaks_low = pickle.load(f)[::2]
    with open('15ff_peaks_high_density.pickle', 'rb') as f:
        fifteen_ff_peaks_high = pickle.load(f)[::2]

    with open('20ff_peaks_low_density.pickle', 'rb') as f:
        twenty_ff_peaks_low = pickle.load(f)[::2]
    with open('20ff_peaks_high_density.pickle', 'rb') as f:
        twenty_ff_peaks_high = pickle.load(f)[::2]
    if measurement_type == '05ff':
        keys = list(d.keys())[::2]
        # keys.append(3.1)
        keys = sorted(keys)
    elif measurement_type == '10ff':
        keys = list(d.keys())[::2]
        # keys.append(3.9)
        keys = sorted(keys)
    elif measurement_type == '15ff':
        keys = list(d.keys())[::2]
        # keys.append(3.3)
        keys = sorted(keys)
    else:
        keys = list(d.keys())[::2]
        # keys.append(3.8)
        keys = sorted(keys)

    d_subset = {k: v for k, v in d.items() if k in keys}
    joy_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in list(d_subset.items())]))
    jfig, jaxes = joyplot(joy_df, colormap=cm.get_cmap('Spectral', 15), fade=True, legend=True, bins=500)
    for jax in jaxes:
        jax.set_xlim(0.0, 30.0)

    jaxes[-1].set_xticks(numpy.arange(0.0, 30.0, 2.0))
    jaxes[-1].set_xticklabels(numpy.arange(0.0, 30.0, 2.0))
    jaxes[-1].set_xlabel('Tb [s]')
    jaxes[2].set_title(measurement_type)
    plt.savefig(name + '/comparison_{}_noinputexperimental_without_peaks.png'.format(measurement_type))
    plt.clf()


def plot_spiketimes(group, name):
    plots(group, 'simulation_results', name)


def spike_variance(exp_results):
    collective_timeseries = {}
    collective_std_div_mean = {}
    fig, axes = plt.subplots()
    for k, v in exp_results.items():
        for simulation in v:
            running_stds = []
            running_means = []
            spikes = simulation.get_burst_data()
            xs = [(t * simulation.timestepsize) for t in list(spikes.keys())]
            ys = list(spikes.values())
            windowsize = 3000
            for i in range(int(len(ys) / windowsize)):
                i = i * windowsize
                slice = ys[i:i+windowsize]
                running_stds.append(numpy.std(slice))
                running_means.append(numpy.mean(slice))
            beta = simulation.beta
            if collective_timeseries.get(beta) is not None:
                collective_timeseries[beta].extend(list(zip(xs, ys)))
            else:
                collective_timeseries[beta] = list(zip(xs, ys))
            std_means = zip(running_stds, running_means)
            if collective_std_div_mean.get(beta) is not None:
                collective_std_div_mean[beta].extend(list(std_means))
            else:
                collective_std_div_mean[beta] = list(std_means)
    colormap = cm.get_cmap('plasma', len(collective_std_div_mean.keys()))
    beta_colorindex_map = {}
    axes.set_xscale('log')
    axes.set_yscale('log')
    for i,k in enumerate(collective_std_div_mean.keys()):
        beta_colorindex_map[k] = i
    all_xs = []
    all_ys = []
    for beta, std_means_at_beta in collective_std_div_mean.items():
        _stds = [s[0] for s in std_means_at_beta]
        _means = [m[1] for m in std_means_at_beta]
        axes.scatter(_means, _stds, color=colormap.colors[beta_colorindex_map[beta]], label=round(beta, 4))
        all_xs.extend(_means)
        all_ys.extend(_stds)

    axes.set_xlabel('<N>')
    axes.set_ylabel(r'$\sigma_N$')

    all_xs = sorted(all_xs)
    all_ys = sorted(all_ys)
    x_bound_1 = 0.515
    y_bound_1 = 1.18
    log_x_regime_1 = [numpy.log(a) for a in all_xs if a < x_bound_1]
    log_y_regime_1 = [numpy.log(b) for b in all_ys if b < y_bound_1]
    log_x_regime_2 = [numpy.log(c) for c in all_xs if c >= x_bound_1]
    log_y_regime_2 = [numpy.log(d) for d in all_ys if d >= y_bound_1]
    m_1, c_1 = numpy.polyfit(log_x_regime_1, log_y_regime_1, 1)  # fit log(y) = m*log(x) + c
    y_fit_regime_1 = [numpy.exp(m_1 * x + c_1) for x in log_x_regime_1]  # calculate the fitted values of y
    m_2, c_2 = numpy.polyfit(log_x_regime_2, log_y_regime_2, 1)  # f   it log(y) = m*log(x) + c
    y_fit_regime_2 = [numpy.exp(m_2 * x + c_2) for x in log_x_regime_2]
    axes.plot([x for x in all_xs if x < x_bound_1], y_fit_regime_1, ':', lw=4, color='black', label='m={}'.format(m_1))
    axes.plot([y for y in all_xs if y >= x_bound_1], y_fit_regime_2, ':', lw=4, color='green', label='m={}'.format(m_2))

    plt.legend(loc=2, ncol=3, prop={'size': 6})
    plt.show()


def spike_analysis(exp_results, argv):
    individual_interspikes = {}
    inputs = draw_from_input_distribution(num_draws=1000)
    for k, v in exp_results.items():
        for simulation in v:
            _isis = simulation.swarm_interburst_dist()
            isis = [i * simulation.timestepsize for i in _isis]
            isis = [i for i in isis]
            beta = simulation.beta
            if individual_interspikes.get(beta) is not None:
                individual_interspikes[beta].extend(isis)
            else:
                individual_interspikes[beta] = isis
    ordered_spiketimes = collections.OrderedDict(sorted(individual_interspikes.items()))

    x = []
    y = []
    y_maybes = []
    datafile = 'experimental_data/experimental/ibs20ff.csv'
    with open(datafile, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            y_maybes.append(y_maybe)
            if y_maybe > 1.5:
                y.append(y_maybe)
    plot_spiketimes(ordered_spiketimes, '20ff')


def load_pickles(ff_count):
    if ff_count == 5:
        ff = '05'
    else:
        ff = ff_count
    data_path = 'simulation_results/{}ff.pickle/'.format(ff)

    with open(data_path + 'beta_sweep_swarm.pickle', 'rb') as f_g:
        d = pickle.load(f_g)
        data = {}
        for k in d.keys():
            new_key = k.split('_')[0]
            data[new_key] = d[k]
        del d
    return data


def spike_analysis_diff(ff_count):
    individual_interspikes = {}
    inputs = draw_from_input_distribution(num_draws=1000)
    data = load_pickles(ff_count)
    if ff_count == 5:
        ff_count = '05'
    for k, v in data.items():
        for beta, _isis in v.items():
            beta = round(float(beta), 4)
            isis = [i * 0.1 for isi_list in _isis for i in isi_list]
            if individual_interspikes.get(beta) is not None:
                individual_interspikes[beta].extend([i for i in isis if i > 1.5])
            else:
                individual_interspikes[beta] = [i for i in isis if i > 1.5]
    ordered_spiketimes = collections.OrderedDict(sorted(individual_interspikes.items()))

    x = []
    y = []
    y_maybes = []
    datafile = 'experimental_data/experimental/ibs{}ff.csv'.format(ff_count, ff_count)
    with open(datafile, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            y_maybes.append(y_maybe)
            if y_maybe > 1.5:
                y.append(y_maybe)

    plot_spiketimes(ordered_spiketimes, '{}ff'.format(ff_count))


def peaks(d, ff_count, ax, t):
    count = int(ff_count)

    ff_color_map = {
        5: 'deepskyblue',
        10: 'darkgreen',
        15: 'firebrick',
        20: 'rebeccapurple'
    }
    epsilon = 0.01
    retdict = {}
    for k in sorted(list(d.keys())):
        vals = sorted(numpy.array(d[k]))
        gkde_vals = gaussian_kde(vals)
        probs = gkde_vals(numpy.linspace(0, 20, 500))
        hist, bin_edges = numpy.histogram(vals, bins=numpy.linspace(0, 20, 500), density=True)
        bin_edges = bin_edges[1:]

        order = 50
        pks, pk_data = find_peaks(probs, height=epsilon, prominence=0.005, distance=order)
        low_peaks = []
        high_peaks = []
        low_peak_heights = []
        high_peak_heights = []
        for pk in pks:
            if bin_edges[pk] < 3.5:
                low_peaks.append(bin_edges[pk])
                low_peak_heights.append(probs[pk])
            else:
                high_peaks.append(bin_edges[pk])
                high_peak_heights.append(probs[pk])
        if len(low_peak_heights) == 0:
            max_low_peak_height = epsilon
        else:
            max_low_peak_height = 0
            for i, ph in enumerate(low_peak_heights):
                if ph > max_low_peak_height:
                    max_low_peak_height = ph
                    min_peak = low_peaks[i]
            if max_low_peak_height < epsilon:
                max_low_peak_height = epsilon

        if len(high_peak_heights) == 0:
            max_high_peak_height = epsilon
        else:
            max_high_peak_height = 0
            for i, ph in enumerate(high_peak_heights):
                if ph > max_high_peak_height:
                    max_high_peak_height = ph
            if max_high_peak_height < epsilon:
                max_high_peak_height = epsilon
        ax.scatter(k, max_high_peak_height / max_low_peak_height, color=ff_color_map[count], marker='o', s=25)
        retdict[k] = max_high_peak_height / max_low_peak_height

    import matplotlib.patches as mpatches
    patches = []
    patches.append(mpatches.Patch(color=ff_color_map[count], label='N={}'.format(count)))
    return retdict


def spike_analysis_from_local(ff_count, ax, t):
    individual_interspikes = {}
    data = load_pickles(ff_count)
    if ff_count == 5:
        ff_count = '05'
    for k, v in data.items():
        for beta, _isis in v.items():
            beta = round(float(beta), 4)
            isis = [i * 0.1 for isi_list in _isis for i in isi_list]
            if individual_interspikes.get(beta) is not None:
                individual_interspikes[beta].extend([i for i in isis if i > 1.5])
            else:
                individual_interspikes[beta] = [i for i in isis if i > 1.5]
    ordered_spiketimes = collections.OrderedDict(sorted(individual_interspikes.items()))

    x = []
    y = []
    y_maybes = []
    datafile = 'experimental_data/experimental/ibs{}ff.csv'.format(ff_count, ff_count)
    with open(datafile, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            y_maybes.append(y_maybe)
            if y_maybe > 1.5:
                y.append(y_maybe)

    patch = peaks(ordered_spiketimes,ff_count, ax, t)
    return patch
