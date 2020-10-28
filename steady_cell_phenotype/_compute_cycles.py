import tempfile
import time
from collections import defaultdict
from typing import Tuple, Callable

import matplotlib
import matplotlib.pyplot as plt
import numba
from flask import Markup, make_response

from equation_system import EquationSystem
from ._util import *

matplotlib.use('agg')


def compute_cycles(model_state, knockouts, continuous, num_iterations, visualize_variables: Dict[str, bool]):
    equation_system = EquationSystem.from_text(model_state['model'])
    equation_system = equation_system.continuous_functional_system(
        continuous_vars=tuple([var for var in continuous if continuous[var]]))
    equation_system = equation_system.knockout_system(knockouts)

    visualized_variables: Tuple[str] = tuple(var for var in visualize_variables if visualize_variables[var])

    # create an update function
    variables: List[str]
    update_fn: Callable
    variables, update_fn = equation_system.as_numpy()
    update_fn = numba.jit(update_fn)

    # associate variable names with their index in vectors
    variable_idx: Dict[str, int] = dict(zip(variables, range(len(variables))))

    constants: List[str] = equation_system.constant_variables()
    constants_vals: Dict[str, int] = {const: int(equation_system[const]) for const in constants}

    # frozenset keys are limit sets
    limit_cycles: Dict[frozenset, List] = dict()  # record limit sets with their ordering. i.e. as cycles

    # phased trajectories include a portion of the limit cycle, up to a fixed element
    # the fixed element is defined to be the first element of the cycle in `limit_cycles`
    phased_trajectory_lengths: Dict[frozenset, BinCounter] = \
        defaultdict(lambda: BinCounter())  # key: limit sets
    phased_limit_set_stats: Dict[Tuple[frozenset, str, int], StreamingStats] = \
        defaultdict(lambda: StreamingStats())

    # decide if we will perform a complete state space search or not
    state_space_size = 3 ** (len(variables) - len(constants))
    perform_complete_search = state_space_size <= num_iterations

    if perform_complete_search:
        state_generator = complete_search_generator(variables=variables,
                                                    constants_vals=constants_vals)
    else:
        state_generator = random_search_generator(num_iterations=num_iterations,
                                                  variables=variables,
                                                  constants_vals=constants_vals)

    for init_state in state_generator:
        state = init_state
        trajectory = list()
        trajectory_set = set()  # set lookup should be faster

        # sim_start = time.time()
        t_state = HashableNdArray(state)  # apparently, conversion from ndarray to tuple is _slow_
        while t_state not in trajectory_set:
            trajectory.append(t_state)
            trajectory_set.add(t_state)
            state = update_fn(state)
            t_state = HashableNdArray(state)

        # sim_end = time.time()

        # separate trajectory into in-bound and limit-cycle parts
        repeated_state = tuple(state)
        repeated_state_index = trajectory.index(repeated_state)
        limit_cycle = trajectory[repeated_state_index:]
        limit_set = frozenset(limit_cycle)

        if limit_set not in limit_cycles:
            limit_cycles[limit_set] = limit_cycle  # record ordering

        # get trajectory with phase
        phase_idx: int = trajectory.index(limit_cycles[limit_set][0])
        phased_trajectory = trajectory[:phase_idx]

        # process_1_end = time.time()

        # record stats
        phased_trajectory_lengths[limit_set].add(len(phased_trajectory))
        for idx, var in itertools.product(range(len(phased_trajectory)), visualized_variables):
            phased_limit_set_stats[(limit_set,
                                    var,
                                    len(phased_trajectory) - 1 - idx)] \
                .add(phased_trajectory[idx][variable_idx[var]])

        # record_1_end = time.time()
        # print("*" * max(0, int(-np.log(sim_end - sim_start))), '\t',
        #       "*" * max(0, int(-np.log(process_1_end - sim_end))), '\t',
        #       "*" * max(0, int(-np.log(record_1_end - process_1_end))))

    # give it a name
    limit_sets = limit_cycles.keys()

    # summary stat
    counts = {limit_set: np.sum(bin_counts.bins) for limit_set, bin_counts in phased_trajectory_lengths.items()}

    # turn off defaults, from defaultdict
    phased_limit_set_stats = dict(phased_limit_set_stats)

    print('simulation complete')

    # create images for each variable
    limit_set_stats_images = {limit_set: dict() for limit_set in limit_sets}
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        plt.rcParams['svg.fonttype'] = 'none'
        for limit_set, var in itertools.product(limit_sets, visualized_variables):
            print(limit_set, var)

            cycle = limit_cycles[limit_set]
            var_idx = variable_idx[var]

            phased_means = np.array(
                [phased_limit_set_stats[(limit_set, var, idx)].mean
                 for idx in range(phased_trajectory_lengths[limit_set].max)])
            phased_means = np.flip(phased_means)

            phased_stdevs = np.array(
                [np.sqrt(phased_limit_set_stats[(limit_set, var, idx)].var)
                 for idx in range(phased_trajectory_lengths[limit_set].max)])
            phased_stdevs = np.flip(phased_stdevs)

            plt.figure(figsize=(6, 4))

            # means, since phased will fix on single value at end
            plt.plot(list(phased_means) + [cycle[0][var_idx]], color='#1f3d87')
            plt.fill_between(range(len(phased_means) + 1),
                             list(phased_means - phased_stdevs) + [cycle[0][var_idx]],
                             list(phased_means + phased_stdevs) + [cycle[0][var_idx]],
                             color='grey',
                             alpha=0.25)
            # dashed cycle in `converging` region
            plt.plot(range(len(phased_means) - len(cycle), len(phased_means) + 1),
                     [cycle[idx][var_idx] for idx in range(len(cycle))] + [cycle[0][var_idx]],
                     color='#F24F00', linestyle='dashed')
            # solid cycle in `converged` region
            plt.plot(range(len(phased_means), len(phased_means) + len(cycle) + 1),
                     [cycle[idx][var_idx] for idx in range(len(cycle))] + [cycle[0][var_idx]],
                     color='#F24F00')

            plt.title(f'Phased Envelope of trajectories for {var}')

            plt.xlabel('Distance to phase-fixing point on limit cycle')
            plt.xticks(range(len(phased_means) + len(cycle) + 1),
                       list(range(len(phased_means), -1 - len(cycle), -1)),
                       rotation=90)
            plt.xlim([0, len(phased_means) + len(cycle)])
            plt.axvline(len(phased_means), linestyle='dotted', color='grey')
            plt.axvline(len(phased_means) - len(cycle), linestyle='dotted', color='grey')

            plt.ylabel('State')
            plt.yticks([0, 1, 2])
            for y_val in range(3):
                plt.axhline(y=y_val, linestyle='dotted', color='grey', alpha=0.5)
            plt.ylim([0 - 0.1, 2 + 0.1])

            plt.tight_layout()

            image_filename = 'dist-' + str(hash(limit_set)) + '-' + var + '.svg'
            plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_dir_name + '/' + image_filename, 'r') as image:
                limit_set_stats_images[limit_set][var] = Markup(image.read())

    print('var images complete')

    # cycle list sorted by frequency
    cycle_list = list(limit_cycles.values())
    cycle_list.sort(key=lambda cycle: counts[frozenset(cycle)], reverse=True)

    # tally of number of cycles of each length, sorted by cycle length
    cycle_len_counts = defaultdict(lambda: 0)
    for cycle in cycle_list:
        cycle_len_counts[len(cycle)] += 1
    cycle_len_counts = tuple(sorted(cycle_len_counts.items()))

    length_distribution_images = dict()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # create length distribution plots
        plt.rcParams['svg.fonttype'] = 'none'
        for limit_set in limit_sets:
            length_dist = phased_trajectory_lengths[limit_set].trimmed_bins()
            plt.figure(figsize=(4, 3))
            plt.bar(x=range(len(length_dist)),
                    height=length_dist,
                    color='#002868')
            plt.title('Distribution of lengths of paths')
            plt.xlabel('Length of path')
            plt.ylabel('Number of states')
            plt.tight_layout()
            image_filename = 'dist-' + str(hash(limit_set)) + '.svg'
            plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_dir_name + '/' + image_filename, 'r') as image:
                length_distribution_images[limit_set] = Markup(image.read())
    print(cycle_list)
    cycles = [{'states': cycle,
               'len': len(cycle),
               'count': counts[frozenset(cycle)],
               'percent': 100 * counts[frozenset(cycle)] / num_iterations,
               'len-dist-image': length_distribution_images[frozenset(cycle)]
               if frozenset(cycle) in length_distribution_images else "",
               'limit-set-stats-images': limit_set_stats_images[frozenset(cycle)]
               if frozenset(cycle) in limit_set_stats_images else dict(), }
              for cycle in cycle_list]
    # respond with the results-of-computation page
    response = make_response(render_template('compute-cycles.html',
                                             cycles=cycles,
                                             variables=variables,
                                             cycle_len_counts=cycle_len_counts,
                                             complete_results=perform_complete_search))
    return response_set_model_cookie(response, model_state)

# def compute_cycles(model_state, knockouts, knockout_model, variables, continuous, num_iterations):
#     """ Run the cycle finding simulation """
#     # TODO: better integrating, thinking more about security
#     with tempfile.TemporaryDirectory() as tmp_dir_name:
#         with open(tmp_dir_name + '/model.txt', 'w') as model_file:
#             model_file.write(knockout_model)
#             model_file.write('\n')
#
#         non_continuous_vars = [variable for variable in variables if not continuous[variable]]
#         if len(non_continuous_vars) > 0:
#             continuity_params = ['-c', '-comit'] + [variable for variable in variables
#                                                     if not continuous[variable]]
#         else:
#             continuity_params = ['-c']
#
#         # randomized search is kind of silly if you ask for more iterations than there are actual states
#         # so go to the complete search mode in this case.
#         if 3 ** len(variables) <= num_iterations:
#             complete_search_params = ['-complete_search']
#         else:
#             complete_search_params = []
#
#         convert_to_c_process = \
#             subprocess.run([os.getcwd() + '/convert.py', '-sim',
#                             '--count', str(num_iterations),
#                             '-i', tmp_dir_name + '/model.txt',
#                             '-o', tmp_dir_name + '/model.c'] + continuity_params + complete_search_params,
#                            capture_output=True)
#         if convert_to_c_process.returncode != 0:
#             response = make_response(error_report(
#                 'Error running converter!\n{}\n{}'.format(html_encode(convert_to_c_process.stdout),
#                                                           html_encode(convert_to_c_process.stderr))))
#             return response_set_model_cookie(response, model_state)
#
#         # copy the header files over
#         subprocess.run(['cp', os.getcwd() + '/mod3ops.h', tmp_dir_name])
#         subprocess.run(['cp', os.getcwd() + '/bloom-filter.h', tmp_dir_name])
#         subprocess.run(['cp', os.getcwd() + '/cycle-table.h', tmp_dir_name])
#         subprocess.run(['cp', os.getcwd() + '/length-count-array.h', tmp_dir_name])
#         # not needed (except for graph simulator)
#         # subprocess.run(['cp', os.getcwd() + '/link-table.h', tmp_dir_name])
#
#         # be fancy about compiler selection
#         installed_compilers = [shutil.which('clang'), shutil.which('gcc'), shutil.which('cc')]
#         compiler = installed_compilers[0] if installed_compilers[0] is not None \
#             else installed_compilers[1] if installed_compilers[1] is not None \
#             else installed_compilers[2]
#
#         compilation_process = \
#             subprocess.run([compiler, '-O3', tmp_dir_name + '/model.c', '-o', tmp_dir_name + '/model'],
#                            capture_output=True)
#         if compilation_process.returncode != 0:
#             with open(tmp_dir_name + '/model.c', 'r') as source_file:
#                 response = make_response(error_report(
#                     'Error running compiler!\n{}\n{}\n{}'.format(html_encode(compilation_process.stdout),
#                                                                  html_encode(compilation_process.stderr),
#                                                                  html_encode(source_file.read()))))
#             return response_set_model_cookie(response, model_state)
#
#         simulation_process = \
#             subprocess.run([tmp_dir_name + '/model'], capture_output=True)
#         if simulation_process.returncode != 0:
#             response = make_response(error_report(
#                 'Error running simulator!\n{}\n{}'.format(html_encode(simulation_process.stdout),
#                                                           html_encode(simulation_process.stderr))))
#             return response_set_model_cookie(response, model_state)
#
#         try:
#             simulator_output = json.loads(simulation_process.stdout.decode())
#         except json.decoder.JSONDecodeError:
#             response = make_response(error_report(
#                 'Error decoding simulator output!\n{}\n{}'.format(html_encode(simulation_process.stdout),
#                                                                   html_encode(simulation_process.stderr))))
#             return response_set_model_cookie(response, model_state)
#
#         # somewhat redundant data in the two fields, combine them, indexed by id
#         combined_output = {
#             cycle['id']: {'length': cycle['length'],
#                           'count': cycle['count'],
#                           'percent': cycle['percent'],
#                           'length-dist': cycle['length-dist']}
#             for cycle in simulator_output['counts']}
#         for cycle in simulator_output['cycles']:
#             combined_output[cycle['id']]['cycle'] = cycle['cycle']
#
#         cycle_list = list(combined_output.values())
#         cycle_list.sort(key=lambda cycle: cycle['count'], reverse=True)
#
#         from collections import defaultdict
#         cycle_len_counts = defaultdict(lambda: 0)
#         for cycle in cycle_list:
#             cycle_len_counts[cycle['length']] += 1
#         cycle_len_counts = tuple(sorted(cycle_len_counts.items()))
#
#         # create length distribution plots
#         plt.rcParams['svg.fonttype'] = 'none'
#         for cycle in combined_output:
#             length_dist = combined_output[cycle]['length-dist']
#             plt.figure(figsize=(4, 3))
#             plt.bar(x=range(len(length_dist)),
#                     height=length_dist,
#                     color='#002868')
#             plt.title('Distribution of lengths of paths')
#             plt.xlabel('Length of path')
#             plt.ylabel('Number of states')
#             plt.tight_layout()
#             image_filename = 'dist-' + str(cycle) + '.svg'
#             plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
#             plt.close()
#             with open(tmp_dir_name + '/' + image_filename, 'r') as image:
#                 combined_output[cycle]['image'] = Markup(image.read())
#
#     performed_complete_search = simulator_output['complete_search'] if 'complete_search' in simulator_output else False
#
#     # respond with the results-of-computation page
#     response = make_response(render_template('compute-cycles.html',
#                                              cycles=cycle_list,
#                                              cycle_len_counts=cycle_len_counts,
#                                              complete_results=performed_complete_search))
#     return response_set_model_cookie(response, model_state)
