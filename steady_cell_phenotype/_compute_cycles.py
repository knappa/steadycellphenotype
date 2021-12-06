import functools
import itertools
import math
import tempfile
from collections import defaultdict
from math import ceil
from typing import Dict, Iterator, List, NamedTuple, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathos
from flask import Markup, make_response, render_template

from steady_cell_phenotype import ParseError
from steady_cell_phenotype._util import (BinCounter, HashableNdArray, batcher,
                                         complete_search_generator,
                                         error_report, get_phased_trajectory,
                                         process_model_text,
                                         random_search_generator)

matplotlib.use("agg")


class TrajectoryStatistics(NamedTuple):
    trajectory_length_counts: Dict[HashableNdArray, BinCounter]
    data_counts: Dict[HashableNdArray, np.ndarray]
    means: Dict[HashableNdArray, np.ndarray]
    variances: Dict[HashableNdArray, np.ndarray]

    @classmethod
    def make(cls, num_variables: int):
        return TrajectoryStatistics(
            trajectory_length_counts=defaultdict(lambda: BinCounter()),
            data_counts=defaultdict(lambda: np.zeros(shape=0, dtype=int)),
            means=defaultdict(lambda: np.zeros(shape=(0, num_variables))),
            variances=defaultdict(lambda: np.zeros(shape=(0, num_variables))),
        )


def batch_trajectory_process(batch, update_fn) -> TrajectoryStatistics:
    num_samples = batch.shape[0]
    num_variables = batch.shape[1]

    trajectory_statistics = TrajectoryStatistics.make(num_variables)
    # unpack
    trajectory_length_counts = trajectory_statistics.trajectory_length_counts
    data_counts = trajectory_statistics.data_counts
    means = trajectory_statistics.means
    scaled_variances = trajectory_statistics.variances

    for batch_idx in range(num_samples):
        trajectory, phase_state = get_phased_trajectory(batch[batch_idx, :], update_fn)
        trajectory_length_counts[phase_state].add(len(trajectory))

        # extract arrays, to reduce dict lookup again
        data_count = data_counts[phase_state]
        mean = means[phase_state]
        scaled_variance = scaled_variances[phase_state]

        old_len: int = data_count.shape[0]
        trajectory_len: int = trajectory.shape[0]

        if trajectory_len == 0:
            continue

        # resize, if necessary
        if old_len < trajectory_len:
            # extend data_count
            new_data_count: np.ndarray = np.zeros(shape=trajectory_len, dtype=int)
            new_data_count[:old_len] = data_count
            data_count: np.ndarray = new_data_count
            data_counts[phase_state] = new_data_count
            # extend means
            new_mean: np.ndarray = np.zeros(shape=(trajectory_len, num_variables))
            new_mean[:old_len, :] = mean
            mean = new_mean
            means[phase_state] = new_mean
            # extend scaled variance
            new_scaled_variance: np.ndarray = np.zeros(
                shape=(trajectory_len, num_variables)
            )
            new_scaled_variance[:old_len] = scaled_variance
            scaled_variance = new_scaled_variance
            scaled_variances[phase_state] = new_scaled_variance

        # do it once
        reversed_trajectory = trajectory[::-1, :]

        # Welford's online algorithm for mean and variance calculation. See Knuth Vol 2, pg 232
        data_count[:trajectory_len] += 1
        old_mean = np.array(mean[:trajectory_len])  # copy
        mean[:trajectory_len] += (reversed_trajectory - old_mean) / np.expand_dims(
            data_count[:trajectory_len], axis=-1
        )
        scaled_variance[:trajectory_len] += (reversed_trajectory - old_mean) * (
            reversed_trajectory - mean[:trajectory_len]
        )

    # need to create a new tuple, as means and scaled variances can get clobbered.
    return TrajectoryStatistics(
        trajectory_length_counts=trajectory_length_counts,
        data_counts=data_counts,
        means=means,
        variances=scaled_variances,
    )


def reducer(
    init_stats: TrajectoryStatistics, new_stats: TrajectoryStatistics
) -> TrajectoryStatistics:
    # unpack
    trajectory_length_counts: Dict[HashableNdArray, BinCounter]
    data_counts: Dict[HashableNdArray, np.ndarray]
    means: Dict[HashableNdArray, np.ndarray]
    variances: Dict[HashableNdArray, np.ndarray]
    trajectory_length_counts, data_counts, means, variances = init_stats

    # unpack
    next_trajectory_length_counts: Dict[HashableNdArray, BinCounter]
    next_data_counts: Dict[HashableNdArray, np.ndarray]
    next_means: Dict[HashableNdArray, np.ndarray]
    next_scaled_variances: Dict[HashableNdArray, np.ndarray]
    (
        next_trajectory_length_counts,
        next_data_counts,
        next_means,
        next_scaled_variances,
    ) = new_stats

    for phase_point in next_trajectory_length_counts.keys():
        num_variables = next_means[phase_point].shape[1]
        if phase_point not in trajectory_length_counts:
            # just copy it
            trajectory_length_counts[phase_point] = next_trajectory_length_counts[
                phase_point
            ]
            data_counts[phase_point] = next_data_counts[phase_point]
            means[phase_point] = next_means[phase_point]
            # remember to scale!
            variances[phase_point] = next_scaled_variances[
                phase_point
            ] / np.expand_dims(data_counts[phase_point], axis=-1)
        else:
            # merge
            trajectory_length_counts[phase_point].add(
                next_trajectory_length_counts[phase_point]
            )
            new_max_len = trajectory_length_counts[phase_point].max

            # data counts
            old_data_count: np.ndarray = data_counts[phase_point]
            next_data_count: np.ndarray = next_data_counts[phase_point]
            new_data_counts = np.zeros(shape=new_max_len, dtype=int)
            new_data_counts[: old_data_count.shape[0]] = old_data_count
            new_data_counts[: next_data_count.shape[0]] += next_data_count

            data_counts[phase_point] = new_data_counts

            # means
            old_mean: np.ndarray = means[phase_point]
            next_mean: np.ndarray = next_means[phase_point]
            new_mean: np.ndarray = np.zeros(
                shape=(new_max_len, num_variables), dtype=np.float64
            )
            max_overlap = min(old_mean.shape[0], next_mean.shape[0])
            # weighted average on the overlap, copy over on the tail
            new_mean[:max_overlap] = (
                np.expand_dims(old_data_count[:max_overlap], axis=-1)
                * old_mean[:max_overlap]
                + np.expand_dims(next_data_count[:max_overlap], axis=-1)
                * next_mean[:max_overlap]
            ) / np.expand_dims(new_data_counts[:max_overlap], axis=-1)
            if old_mean.shape[0] > next_mean.shape[0]:
                new_mean[max_overlap:, :] = old_mean[max_overlap:, :]
            else:
                new_mean[max_overlap:, :] = next_mean[max_overlap:, :]

            means[phase_point] = new_mean

            # scaled variances
            old_variance: np.ndarray = variances[phase_point]
            next_scaled_variance: np.ndarray = next_scaled_variances[phase_point]
            new_variance: np.ndarray = np.zeros(
                shape=(new_max_len, num_variables), dtype=np.float64
            )
            # "weighted average" on the overlap, copy over on the tail
            old_proportion = np.expand_dims(
                old_data_count[:max_overlap] / new_data_counts[:max_overlap], axis=-1
            )
            next_proportion = np.expand_dims(
                next_data_count[:max_overlap] / new_data_counts[:max_overlap], axis=-1
            )
            new_variance[:max_overlap] = (
                old_proportion * old_variance[:max_overlap]
                + next_proportion
                * (
                    next_scaled_variance[:max_overlap]
                    / np.expand_dims(next_data_count[:max_overlap], axis=-1)
                )
                + ((old_mean[:max_overlap] - next_mean[:max_overlap]) ** 2)
                * old_proportion
                * next_proportion
            )
            if old_variance.shape[0] > next_scaled_variance.shape[0]:
                new_variance[max_overlap:, :] = old_variance[max_overlap:, :]
            else:
                new_variance[max_overlap:, :] = next_scaled_variance[
                    max_overlap:, :
                ] / np.expand_dims(next_data_count[max_overlap:], axis=-1)

            variances[phase_point] = new_variance

    return TrajectoryStatistics(trajectory_length_counts, data_counts, means, variances)


def compute_cycles(
    *,
    model_text: str,
    knockouts: Dict[str, int],
    continuous: Dict[str, bool],
    num_iterations: int,
    visualize_variables: Dict[str, bool],
):
    """

    Parameters
    ----------
    model_text
    knockouts
    continuous
    num_iterations
    visualize_variables

    Returns
    -------

    """
    # turn boolean-valued dictionary to list of variables to visualize
    visualized_variables: Tuple[str] = tuple(
        var for var in visualize_variables if visualize_variables[var]
    )

    # create an update function and equation system
    try:
        variables, update_fn, equation_system = process_model_text(
            model_text, knockouts, continuous
        )
    except ParseError as e:
        return make_response(error_report(e.message))

    # compile and warm up the jitter
    # update_fn = numba.jit(update_fn)
    # update_fn(np.zeros(len(variables), dtype=np.int64))

    # associate variable names with their index in vectors
    variable_idx: Dict[str, int] = dict(zip(variables, range(len(variables))))

    constants: List[str] = equation_system.constant_variables()
    constants_vals: Dict[str, int] = {
        const: int(equation_system[const]) for const in constants
    }

    # decide if we will perform a complete state space search or not
    state_space_size = 3 ** (len(variables) - len(constants))
    perform_complete_search = state_space_size <= num_iterations

    state_generator: Iterator[np.ndarray]
    if perform_complete_search:
        state_generator = complete_search_generator(
            variables=variables, constants_vals=constants_vals
        )
        num_iterations = state_space_size
    else:
        state_generator = random_search_generator(
            num_iterations=num_iterations,
            variables=variables,
            constants_vals=constants_vals,
        )

    max_threads = max(
        1,
        min(
            pathos.multiprocessing.cpu_count() - 1, math.floor(state_space_size / 1000)
        ),
    )

    batch_generator = batcher(
        state_generator, variables, batch_size=min(1000, num_iterations // max_threads)
    )

    trajectory_length_counts: Dict[HashableNdArray, BinCounter] = dict()
    data_counts: Dict[HashableNdArray, np.ndarray] = dict()
    means: Dict[HashableNdArray, np.ndarray] = dict()
    variances: Dict[HashableNdArray, np.ndarray] = dict()
    with pathos.multiprocessing.ProcessPool(nodes=max_threads) as pool:
        trajectory_length_counts, data_counts, means, variances = functools.reduce(
            reducer,
            pool.uimap(
                batch_trajectory_process, batch_generator, itertools.repeat(update_fn)
            ),
            TrajectoryStatistics(
                trajectory_length_counts, data_counts, means, variances
            ),
        )

    phase_points = trajectory_length_counts.keys()

    # recreate limit cycles from phase point keys
    limit_cycles: Dict[
        HashableNdArray, List
    ] = dict()  # record limit sets with their ordering. i.e. as cycles
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
        plt.rcParams["svg.fonttype"] = "none"
        for phase_point, var in itertools.product(phase_points, visualized_variables):
            cycle = limit_cycles[phase_point]
            var_idx = variable_idx[var]

            var_means = np.flipud(means[phase_point][:, var_idx])
            var_stdevs = np.flipud(np.sqrt(variances[phase_point][:, var_idx]))

            plt.figure(figsize=(6, 4))

            # means, since phased will fix on single value at end
            plt.plot(list(var_means) + [cycle[0][var_idx]], color="#1f3d87")
            plt.fill_between(
                range(len(var_means) + 1),
                list(var_means - var_stdevs) + [cycle[0][var_idx]],
                list(var_means + var_stdevs) + [cycle[0][var_idx]],
                color="grey",
                alpha=0.25,
            )
            # dashed cycle in `converging` region
            plt.plot(
                range(len(var_means) - len(cycle), len(var_means) + 1),
                [cycle[idx][var_idx] for idx in range(len(cycle))]
                + [cycle[0][var_idx]],
                color="#F24F00",
                linestyle="dashed",
            )
            # solid cycle in `converged` region
            plt.plot(
                range(len(var_means), len(var_means) + len(cycle) + 1),
                [cycle[idx][var_idx] for idx in range(len(cycle))]
                + [cycle[0][var_idx]],
                color="#F24F00",
            )

            plt.title(f"Phased Envelope of trajectories for {var}")

            plt.xlabel("Distance to phase-fixing point on limit cycle")
            max_tick_goal = 40
            tick_gap = max(1, ceil((len(var_means) + len(cycle) + 1) / max_tick_goal))
            x_tick_locations = [
                x
                for x in range(0, len(var_means) + len(cycle) + 1)
                if (x - len(var_means)) % tick_gap == 0
            ]
            x_tick_labels = list(map(lambda x: len(var_means) - x, x_tick_locations))
            plt.xticks(x_tick_locations, x_tick_labels, rotation=90)
            plt.xlim([0, len(var_means) + len(cycle)])
            plt.axvline(len(var_means), linestyle="dotted", color="grey")
            plt.axvline(len(var_means) - len(cycle), linestyle="dotted", color="grey")

            plt.ylabel("State")
            plt.yticks([0, 1, 2])
            for y_val in range(3):
                plt.axhline(y=y_val, linestyle="dotted", color="grey", alpha=0.5)
            plt.ylim([0 - 0.1, 2 + 0.1])

            plt.tight_layout()

            image_filename = "dist-" + str(hash(phase_point)) + "-" + var + ".svg"
            plt.savefig(
                tmp_dir_name + "/" + image_filename, transparent=True, pad_inches=0.0
            )
            plt.close()
            with open(tmp_dir_name + "/" + image_filename, "r") as image:
                limit_set_stats_images[phase_point][var] = Markup(image.read())

    # print('var images complete')

    # cycle list sorted by frequency
    cycle_list = list(limit_cycles.values())
    cycle_list.sort(
        key=lambda cyc: trajectory_length_counts[cyc[0]].total(), reverse=True
    )

    # tally of number of cycles of each length, sorted by cycle length
    cycle_len_counts = defaultdict(lambda: 0)
    for cycle in cycle_list:
        cycle_len_counts[len(cycle)] += 1
    cycle_len_counts = tuple(sorted(cycle_len_counts.items()))

    length_distribution_images = dict()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # create length distribution plots
        plt.rcParams["svg.fonttype"] = "none"
        for phase_point in phase_points:
            length_dist = trajectory_length_counts[phase_point].trimmed_bins()
            plt.figure(figsize=(4, 3))
            plt.bar(x=range(len(length_dist)), height=length_dist, color="#002868")
            plt.title("Distribution of lengths of paths")
            plt.xlabel("Length of path")
            plt.ylabel("Number of states")
            plt.tight_layout()
            image_filename = "dist-" + str(hash(phase_point)) + ".svg"
            plt.savefig(
                tmp_dir_name + "/" + image_filename, transparent=True, pad_inches=0.0
            )
            plt.close()
            with open(tmp_dir_name + "/" + image_filename, "r") as image:
                length_distribution_images[phase_point] = Markup(image.read())

    def get_key(cyc):
        return cyc[0]

    def get_count(cyc):
        return trajectory_length_counts[get_key(cyc)].total()

    cycles = [
        {
            "states": cycle,
            "len": len(cycle),
            "count": get_count(cycle),
            "percent": 100 * get_count(cycle) / num_iterations,
            "len-dist-image": length_distribution_images[get_key(cycle)]
            if get_key(cycle) in length_distribution_images
            else "",
            "limit-set-stats-images": limit_set_stats_images[get_key(cycle)]
            if get_key(cycle) in limit_set_stats_images
            else dict(),
        }
        for cycle in cycle_list
    ]

    # respond with the results-of-computation page
    return make_response(
        render_template(
            "compute-cycles.html",
            cycles=cycles,
            variables=variables,
            cycle_len_counts=cycle_len_counts,
            complete_results=perform_complete_search,
        )
    )
