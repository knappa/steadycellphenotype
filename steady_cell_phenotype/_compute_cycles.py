from collections import defaultdict
import functools
from math import ceil
import tempfile
from typing import Callable, Tuple

from flask import make_response, Markup
import matplotlib
import matplotlib.pyplot as plt
import pathos

from equation_system import EquationSystem
from ._util import *

matplotlib.use('agg')


def get_trajectory(init_state, update_fn) \
        -> Tuple[np.ndarray, HashableNdArray]:
    """
    evolve an initial state until it reaches a limit cycle

    Parameters
    ----------
    init_state
    update_fn

    Returns
    -------
    trajectory, phase-point pair
    """
    state = init_state
    trajectory = list()
    trajectory_set = set()  # set lookup should be faster

    t_state = HashableNdArray(state)  # apparently, conversion from ndarray to tuple is _slow_
    while t_state not in trajectory_set:
        trajectory.append(t_state)
        trajectory_set.add(t_state)
        state = update_fn(state)
        t_state = HashableNdArray(state)

    # separate trajectory into in-bound and limit-cycle parts
    repeated_state = HashableNdArray(state)
    repeated_state_index = trajectory.index(repeated_state)
    limit_cycle = trajectory[repeated_state_index:]

    # find state in limit cycle with smallest hash (i.e. smallest lexicographic
    # ordering if there is no integer overflow)
    # this is our phase fixing point
    cycle_min_index: int = 0
    cycle_min: int = hash(limit_cycle[0])
    for idx in range(1, len(limit_cycle)):
        nxt_hash: int = hash(limit_cycle[idx])
        if nxt_hash < cycle_min:
            cycle_min_index = idx
            cycle_min = nxt_hash

    # get trajectory with phase
    phase_idx: int = len(trajectory) - len(limit_cycle) + cycle_min_index
    phased_trajectory = np.array([hashable.array for hashable in trajectory[:phase_idx]], dtype=np.int64)

    return phased_trajectory, trajectory[phase_idx]


def batch_trajectory_process(batch, update_fn) \
        -> Tuple[Dict[HashableNdArray, BinCounter],
                 Dict[HashableNdArray, np.ndarray],
                 Dict[HashableNdArray, np.ndarray],
                 Dict[HashableNdArray, np.ndarray]]:
    num_samples = batch.shape[0]
    num_variables = batch.shape[1]

    trajectory_length_counts: Dict[HashableNdArray, BinCounter] = \
        defaultdict(lambda: BinCounter())
    data_counts: Dict[HashableNdArray, np.ndarray] = \
        defaultdict(lambda: np.zeros(shape=0, dtype=np.int))
    means: Dict[HashableNdArray, np.ndarray] = \
        defaultdict(lambda: np.zeros(shape=(0, num_variables)))
    scaled_variances: Dict[HashableNdArray, np.ndarray] = \
        defaultdict(lambda: np.zeros(shape=(0, num_variables)))

    for batch_idx in range(num_samples):
        trajectory, phase_state = get_trajectory(batch[batch_idx, :], update_fn)
        trajectory_length_counts[phase_state].add(len(trajectory))

        # extract arrays, to reduce dict lookup again
        data_count = data_counts[phase_state]
        mean = means[phase_state]
        scaled_variance = scaled_variances[phase_state]

        old_len = data_count.shape[0]
        trajectory_len = trajectory.shape[0]

        if trajectory_len == 0:
            continue

        # resize, if necessary
        if old_len < trajectory_len:
            # extend data_count
            new_data_count = np.zeros(shape=trajectory_len, dtype=np.int)
            new_data_count[:old_len] = data_count
            data_count = new_data_count
            data_counts[phase_state] = new_data_count
            # extend means
            new_mean = np.zeros(shape=(trajectory_len, num_variables))
            new_mean[:old_len, :] = mean
            mean = new_mean
            means[phase_state] = new_mean
            # extend scaled variance
            new_scaled_variance = np.zeros(shape=(trajectory_len, num_variables))
            new_scaled_variance[:old_len] = scaled_variance
            scaled_variance = new_scaled_variance
            scaled_variances[phase_state] = new_scaled_variance

        # do it once
        reversed_trajectory = trajectory[::-1, :]

        # Welford's online algorithm for mean and variance calculation. See Knuth Vol 2, pg 232
        data_count[:trajectory_len] += 1
        old_mean = np.array(mean[:trajectory_len])  # copy
        mean[:trajectory_len] += \
            (reversed_trajectory - old_mean) / np.expand_dims(data_count[:trajectory_len], axis=-1)
        scaled_variance[:trajectory_len] += \
            (reversed_trajectory - old_mean) * (reversed_trajectory - mean[:trajectory_len])

    return trajectory_length_counts, data_counts, means, scaled_variances


def reducer(init_stats: Tuple[Dict[HashableNdArray, BinCounter],
                              Dict[HashableNdArray, np.ndarray],
                              Dict[HashableNdArray, np.ndarray],
                              Dict[HashableNdArray, np.ndarray]],
            new_stats: Tuple[Dict[HashableNdArray, BinCounter],
                             Dict[HashableNdArray, np.ndarray],
                             Dict[HashableNdArray, np.ndarray],
                             Dict[HashableNdArray, np.ndarray]]) \
        -> Tuple[Dict[HashableNdArray, BinCounter],
                 Dict[HashableNdArray, np.ndarray],
                 Dict[HashableNdArray, np.ndarray],
                 Dict[HashableNdArray, np.ndarray]]:
    # unpack
    trajectory_length_counts: Dict[HashableNdArray, BinCounter]
    data_counts: Dict[HashableNdArray, np.ndarray]
    means: Dict[HashableNdArray, np.ndarray]
    variances: Dict[HashableNdArray, np.ndarray]
    trajectory_length_counts, data_counts, means, variances = init_stats

    next_trajectory_length_counts: Dict[HashableNdArray, BinCounter]
    next_data_counts: Dict[HashableNdArray, np.ndarray]
    next_means: Dict[HashableNdArray, np.ndarray]
    next_scaled_variances: Dict[HashableNdArray, np.ndarray]
    next_trajectory_length_counts, next_data_counts, next_means, next_scaled_variances = new_stats

    for phase_point in next_trajectory_length_counts.keys():
        num_variables = next_means[phase_point].shape[1]
        if phase_point not in trajectory_length_counts:
            # just copy it
            trajectory_length_counts[phase_point] = next_trajectory_length_counts[phase_point]
            data_counts[phase_point] = next_data_counts[phase_point]
            means[phase_point] = next_means[phase_point]
            # remember to scale!
            variances[phase_point] = \
                next_scaled_variances[phase_point] / np.expand_dims(data_counts[phase_point], axis=-1)
        else:
            # merge
            trajectory_length_counts[phase_point].add(next_trajectory_length_counts[phase_point])
            new_max_len = trajectory_length_counts[phase_point].max

            # data counts
            old_data_count: np.ndarray = data_counts[phase_point]
            next_data_count: np.ndarray = next_data_counts[phase_point]
            new_data_counts = np.zeros(shape=new_max_len, dtype=np.int)
            new_data_counts[:old_data_count.shape[0]] = old_data_count
            new_data_counts[:next_data_count.shape[0]] += next_data_count

            data_counts[phase_point] = new_data_counts

            # means
            old_mean: np.ndarray = means[phase_point]
            next_mean: np.ndarray = next_means[phase_point]
            new_mean: np.ndarray = np.zeros(shape=(new_max_len, num_variables), dtype=np.float)
            max_overlap = min(old_mean.shape[0], next_mean.shape[0])
            # weighted average on the overlap, copy over on the tail
            new_mean[:max_overlap] = \
                (np.expand_dims(old_data_count[:max_overlap], axis=-1) * old_mean[:max_overlap] +
                 np.expand_dims(next_data_count[:max_overlap], axis=-1) * next_mean[:max_overlap]) \
                / np.expand_dims(new_data_counts[:max_overlap], axis=-1)
            if old_mean.shape[0] > next_mean.shape[0]:
                new_mean[max_overlap:, :] = old_mean[max_overlap:, :]
            else:
                new_mean[max_overlap:, :] = next_mean[max_overlap:, :]

            means[phase_point] = new_mean

            # scaled variances
            old_variance: np.ndarray = variances[phase_point]
            next_scaled_variance: np.ndarray = next_scaled_variances[phase_point]
            new_variance: np.ndarray = np.zeros(shape=(new_max_len, num_variables), dtype=np.float)
            # "weighted average" on the overlap, copy over on the tail
            old_proportion = \
                np.expand_dims(old_data_count[:max_overlap] / new_data_counts[:max_overlap], axis=-1)
            next_proportion = \
                np.expand_dims(next_data_count[:max_overlap] / new_data_counts[:max_overlap], axis=-1)
            new_variance[:max_overlap] = \
                (old_proportion * old_variance[:max_overlap] +
                 next_proportion * (next_scaled_variance[:max_overlap]
                                    / np.expand_dims(next_data_count[:max_overlap], axis=-1)) +
                 ((old_mean[:max_overlap] - next_mean[:max_overlap]) ** 2) *
                 old_proportion * next_proportion)
            if old_variance.shape[0] > next_scaled_variance.shape[0]:
                new_variance[max_overlap:, :] = \
                    old_variance[max_overlap:, :]
            else:
                new_variance[max_overlap:, :] = \
                    next_scaled_variance[max_overlap:, :] / np.expand_dims(next_data_count[max_overlap:], axis=-1)

            variances[phase_point] = new_variance

    return trajectory_length_counts, data_counts, means, variances


def compute_cycles(model_text,
                   knockouts,
                   continuous,
                   num_iterations: int,
                   visualize_variables: Dict[str, bool]):
    equation_system = EquationSystem.from_text(model_text)
    equation_system = equation_system.continuous_functional_system(
        continuous_vars=tuple([var for var in continuous if continuous[var]]))
    equation_system = equation_system.knockout_system(knockouts)

    visualized_variables: Tuple[str] = tuple(var for var in visualize_variables if visualize_variables[var])

    # create an update function
    variables: List[str]
    update_fn: Callable
    variables, update_fn = equation_system.as_numpy()
    update_fn = numba.jit(update_fn)
    # warm up the jitter
    update_fn(np.zeros(len(variables), dtype=np.int))

    # associate variable names with their index in vectors
    variable_idx: Dict[str, int] = dict(zip(variables, range(len(variables))))

    constants: List[str] = equation_system.constant_variables()
    constants_vals: Dict[str, int] = {const: int(equation_system[const]) for const in constants}

    # decide if we will perform a complete state space search or not
    state_space_size = 3 ** (len(variables) - len(constants))
    perform_complete_search = state_space_size <= num_iterations

    state_generator: Iterator[np.ndarray]
    if perform_complete_search:
        state_generator = complete_search_generator(variables=variables,
                                                    constants_vals=constants_vals)
        num_iterations = state_space_size
    else:
        state_generator = random_search_generator(num_iterations=num_iterations,
                                                  variables=variables,
                                                  constants_vals=constants_vals)

    max_threads = max(2, pathos.multiprocessing.cpu_count() - 1)

    batch_generator = batcher(state_generator,
                              variables,
                              batch_size=min(1000, num_iterations // max_threads))

    trajectory_length_counts: Dict[HashableNdArray, BinCounter] = dict()
    data_counts: Dict[HashableNdArray, np.ndarray] = dict()
    means: Dict[HashableNdArray, np.ndarray] = dict()
    variances: Dict[HashableNdArray, np.ndarray] = dict()
    with pathos.multiprocessing.ProcessPool(nodes=max_threads) as pool:
        trajectory_length_counts, data_counts, means, variances = \
            functools.reduce(reducer,
                             pool.uimap(batch_trajectory_process,
                                        batch_generator,
                                        itertools.repeat(update_fn)),
                             (trajectory_length_counts, data_counts, means, variances))

    phase_points = trajectory_length_counts.keys()

    # recreate limit cycles from phase point keys
    limit_cycles: Dict[HashableNdArray, List] = dict()  # record limit sets with their ordering. i.e. as cycles
    for phase_point in phase_points:
        cycle = list()
        t_state = phase_point
        state = phase_point.array
        # do-while
        cycle.append(t_state)
        state = update_fn(state)
        t_state = HashableNdArray(state)
        while t_state != phase_point:
            cycle.append(t_state)
            state = update_fn(state)
            t_state = HashableNdArray(state)
        # record cycle
        limit_cycles[phase_point] = cycle

    # create images for each variable
    limit_set_stats_images = {phase_point: dict() for phase_point in phase_points}
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        plt.rcParams['svg.fonttype'] = 'none'
        for phase_point, var in itertools.product(phase_points, visualized_variables):
            cycle = limit_cycles[phase_point]
            var_idx = variable_idx[var]

            var_means = np.flipud(means[phase_point][:, var_idx])
            var_stdevs = np.flipud(np.sqrt(variances[phase_point][:, var_idx]))

            plt.figure(figsize=(6, 4))

            # means, since phased will fix on single value at end
            plt.plot(list(var_means) + [cycle[0][var_idx]], color='#1f3d87')
            plt.fill_between(range(len(var_means) + 1),
                             list(var_means - var_stdevs) + [cycle[0][var_idx]],
                             list(var_means + var_stdevs) + [cycle[0][var_idx]],
                             color='grey',
                             alpha=0.25)
            # dashed cycle in `converging` region
            plt.plot(range(len(var_means) - len(cycle), len(var_means) + 1),
                     [cycle[idx][var_idx] for idx in range(len(cycle))] + [cycle[0][var_idx]],
                     color='#F24F00', linestyle='dashed')
            # solid cycle in `converged` region
            plt.plot(range(len(var_means), len(var_means) + len(cycle) + 1),
                     [cycle[idx][var_idx] for idx in range(len(cycle))] + [cycle[0][var_idx]],
                     color='#F24F00')

            plt.title(f'Phased Envelope of trajectories for {var}')

            plt.xlabel('Distance to phase-fixing point on limit cycle')
            max_tick_goal = 40
            tick_gap = max(1, ceil((len(var_means) + len(cycle) + 1) / max_tick_goal))
            x_tick_locations = [x for x in range(0, len(var_means) + len(cycle) + 1)
                                if (x - len(var_means)) % tick_gap == 0]
            x_tick_labels = list(map(lambda x: len(var_means) - x,
                                     x_tick_locations))
            plt.xticks(x_tick_locations, x_tick_labels, rotation=90)
            # plt.xticks(range(len(var_means) + len(cycle) + 1),
            #            list(range(len(var_means), -1 - len(cycle), -1)),
            #            rotation=90)
            plt.xlim([0, len(var_means) + len(cycle)])
            plt.axvline(len(var_means), linestyle='dotted', color='grey')
            plt.axvline(len(var_means) - len(cycle), linestyle='dotted', color='grey')

            plt.ylabel('State')
            plt.yticks([0, 1, 2])
            for y_val in range(3):
                plt.axhline(y=y_val, linestyle='dotted', color='grey', alpha=0.5)
            plt.ylim([0 - 0.1, 2 + 0.1])

            plt.tight_layout()

            image_filename = 'dist-' + str(hash(phase_point)) + '-' + var + '.svg'
            plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_dir_name + '/' + image_filename, 'r') as image:
                limit_set_stats_images[phase_point][var] = Markup(image.read())

    # print('var images complete')

    # cycle list sorted by frequency
    cycle_list = list(limit_cycles.values())
    cycle_list.sort(key=lambda cyc: trajectory_length_counts[cyc[0]].total(), reverse=True)

    # tally of number of cycles of each length, sorted by cycle length
    cycle_len_counts = defaultdict(lambda: 0)
    for cycle in cycle_list:
        cycle_len_counts[len(cycle)] += 1
    cycle_len_counts = tuple(sorted(cycle_len_counts.items()))

    length_distribution_images = dict()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # create length distribution plots
        plt.rcParams['svg.fonttype'] = 'none'
        for phase_point in phase_points:
            length_dist = trajectory_length_counts[phase_point].trimmed_bins()
            plt.figure(figsize=(4, 3))
            plt.bar(x=range(len(length_dist)),
                    height=length_dist,
                    color='#002868')
            plt.title('Distribution of lengths of paths')
            plt.xlabel('Length of path')
            plt.ylabel('Number of states')
            plt.tight_layout()
            image_filename = 'dist-' + str(hash(phase_point)) + '.svg'
            plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_dir_name + '/' + image_filename, 'r') as image:
                length_distribution_images[phase_point] = Markup(image.read())

    def get_key(cyc):
        return cyc[0]

    def get_count(cyc):
        return trajectory_length_counts[get_key(cyc)].total()

    cycles = [{'states'                : cycle,
               'len'                   : len(cycle),
               'count'                 : get_count(cycle),
               'percent'               : 100 * get_count(cycle) / num_iterations,
               'len-dist-image'        : length_distribution_images[get_key(cycle)]
               if get_key(cycle) in length_distribution_images else "",
               'limit-set-stats-images': limit_set_stats_images[get_key(cycle)]
               if get_key(cycle) in limit_set_stats_images else dict(), }
              for cycle in cycle_list]
    # respond with the results-of-computation page
    return make_response(render_template('compute-cycles.html',
                                         cycles=cycles,
                                         variables=variables,
                                         cycle_len_counts=cycle_len_counts,
                                         complete_results=perform_complete_search))
