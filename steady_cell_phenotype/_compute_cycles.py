import itertools
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import sympy
from flask import Markup, make_response

from equation_system import EquationSystem
from ._util import *
import numpy as np

matplotlib.use('agg')


def compute_cycles(model_state, knockouts, knockout_model, variables, continuous, num_iterations):
    equation_system = EquationSystem.from_text(model_state['model'])
    # TODO: knockouts, continuity

    # TODO: reimplement complete search
    # # randomized search is kind of silly if you ask for more iterations than there are actual states
    # # so go to the complete search mode in this case.
    # if 3 ** len(variables) <= num_iterations:
    #     complete_search_params = ['-complete_search']
    # else:
    #     complete_search_params = []

    variables, eqns_dict = equation_system.as_sympy()
    variable_idx = dict(zip(variables, range(len(variables))))

    # create an update function
    variables_sympy = tuple(sympy.Symbol(var, integer=True) for var in variables)
    update_fn = sympy.lambdify(variables_sympy, tuple(eqns_dict[var] % 3 for var in variables), modules=['numpy'])

    constants = equation_system.constant_variables()
    constants_dict = {const: int(equation_system[const]) for const in constants}

    # keys are limit cycles, values are trajectories into them
    trajectories = defaultdict(lambda: [])  # key: limit sets
    trajectory_lengths = defaultdict(lambda: [])  # key: limit sets
    counts = defaultdict(lambda: 0)
    limit_cycles = dict()  # record limit sets with their ordering. i.e. as cycles

    # phased trajectories include a portion of the limit cycle, up to a fixed element
    # the fixed element is defined to be the first element of the cycle in `limit_cycles`
    phased_trajectories = defaultdict(lambda: [])
    phased_trajectory_lengths = defaultdict(lambda: [])  # key: limit sets

    for iteration in range(num_iterations):
        # making the choice to start constant variables at their constant values
        init_state_dict = {var: np.random.randint(3) if var not in constants else constants_dict[var]
                           for var in variables}
        state = tuple(init_state_dict[var] for var in variables)

        trajectory = list()
        trajectory_set = set()  # set lookup should be faster
        while state not in trajectory_set:
            trajectory.append(state)
            trajectory_set.add(state)
            state = update_fn(*state)

        # separate trajectory into in-bound and limit-cycle parts
        repeated_state = state
        repeated_state_index = trajectory.index(repeated_state)
        trajectory_to_limit_cycle = trajectory[:repeated_state_index]
        limit_cycle = trajectory[repeated_state_index:]
        limit_set = frozenset(limit_cycle)

        if limit_set not in limit_cycles:
            limit_cycles[limit_set] = limit_cycle  # record ordering

        # record trajectory with phase
        phase_idx = trajectory.index(limit_cycles[limit_set][0])
        phased_trajectory = trajectory[:phase_idx]
        phased_trajectories[limit_set].append(phased_trajectory)
        phased_trajectory_lengths[limit_set].append(len(phased_trajectory))

        counts[limit_set] += 1
        trajectory_lengths[limit_set].append(len(trajectory_to_limit_cycle))
        if len(trajectory_to_limit_cycle) > 0:
            trajectories[limit_set].append(trajectory_to_limit_cycle)

    # give it a name
    limit_sets = trajectories.keys()

    # collect stats
    limit_set_stats = dict()
    for limit_set in limit_sets:
        length_dist = np.bincount(trajectory_lengths[limit_set])  # TODO: should length_dist be phased?
        max_len = len(length_dist) - 1

        # indices: variable name, then max_len - distance to limit set
        state_dataset_by_var = {var: [[] for _ in range(max_len)] for var in variables}
        for trajectory in trajectories[limit_set]:
            for var, idx in itertools.product(variables, range(len(trajectory))):
                state_dataset_by_var[var][-1 - idx].append(trajectory[-1 - idx][variable_idx[var]])

        limit_set_stats[limit_set] = {var: [[] for _ in range(max_len)] for var in variables}
        for var, idx in itertools.product(variables, range(max_len)):
            data = state_dataset_by_var[var][idx]
            limit_set_stats[limit_set][var][idx] = {'mean': np.mean(data),
                                                    'stdev': np.std(data)}
    # collect phased stats
    phased_limit_set_stats = dict()
    for limit_set in limit_sets:
        phased_length_dist = np.bincount(phased_trajectory_lengths[limit_set])
        phased_max_len = len(phased_length_dist) - 1

        # indices: variable name, then max_len - distance to limit set
        phased_state_dataset_by_var = {var: [[] for _ in range(phased_max_len)] for var in variables}
        for trajectory in phased_trajectories[limit_set]:
            for var, idx in itertools.product(variables, range(len(trajectory))):
                phased_state_dataset_by_var[var][-1 - idx].append(trajectory[-1 - idx][variable_idx[var]])

        phased_limit_set_stats[limit_set] = {var: [[] for _ in range(phased_max_len)] for var in variables}
        for var, idx in itertools.product(variables, range(phased_max_len)):
            data = phased_state_dataset_by_var[var][idx]
            phased_limit_set_stats[limit_set][var][idx] = {'mean': np.mean(data),
                                                           'stdev': np.std(data)}

    # create images for each variable
    limit_set_stats_images = {limit_set: dict() for limit_set in limit_sets}
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        plt.rcParams['svg.fonttype'] = 'none'
        for limit_set, var in itertools.product(limit_sets, variables):
            stats = limit_set_stats[limit_set]
            phased_stats = phased_limit_set_stats[limit_set]

            cycle = limit_cycles[limit_set]
            var_idx = variable_idx[var]

            cycle_mean = np.mean([state[var_idx] for state in cycle])
            cycle_stdev = np.std([state[var_idx] for state in cycle])

            means = np.array([stats[var][idx]['mean'] for idx in range(len(stats[var]))])
            stdevs = np.array([stats[var][idx]['stdev'] for idx in range(len(stats[var]))])

            phased_means = np.array([phased_stats[var][idx]['mean'] for idx in range(len(phased_stats[var]))])
            phased_stdevs = np.array([phased_stats[var][idx]['stdev'] for idx in range(len(phased_stats[var]))])

            plt.figure(figsize=(6, 6))

            ###############################################################################
            plt.subplot(2, 1, 1)
            plt.plot(means)
            plt.fill_between(range(len(means)), means - stdevs, means + stdevs, color='grey', alpha=0.25)

            plt.axvline(x=len(means) - 1, color='grey', linestyle='dotted')
            plt.axvline(x=len(means), color='grey', linestyle='dotted')

            mean_interp = np.array([means[-1], cycle_mean, cycle_mean])
            stdev_interp = np.array([stdevs[-1], cycle_stdev, cycle_stdev])
            xvals = np.array([len(means) - 1, len(means), len(means) + len(cycle)])
            plt.plot(xvals,
                     mean_interp,
                     color='grey',
                     linestyle='dashed')
            plt.fill_between(xvals,
                             mean_interp - stdev_interp,
                             mean_interp + stdev_interp,
                             color='grey',
                             alpha=0.25)

            plt.plot(range(len(means), len(means) + len(cycle)),
                     [cycle[idx][var_idx] for idx in range(len(cycle))],
                     color='orange')
            plt.plot([len(means) + len(cycle) - 1, len(means) + len(cycle)], [cycle[-1][var_idx], cycle[0][var_idx]],
                     color='orange', linestyle='dotted')

            plt.title(f'Envelope of trajectories for {var}')

            plt.xlabel('Distance to limit cycle')
            plt.xticks(range(len(means) + len(cycle) + 1),
                       list(range(len(means), 0, -1)) + ([0] * (1 + len(cycle))),
                       rotation=45)
            plt.xlim([0, len(means) + len(cycle)])

            plt.ylabel('State')
            plt.yticks([0, 1, 2])
            for y_val in range(3):
                plt.axhline(y=y_val, linestyle='dotted', color='grey', alpha=0.5)
            plt.ylim([0 - 0.1, 2 + 0.1])

            ###############################################################################
            plt.subplot(2, 1, 2)
            # means, since phased will fix on single value at end
            plt.plot(list(phased_means) + [cycle[0][var_idx]], color='C0')
            plt.fill_between(range(len(phased_means) + 1),
                             list(phased_means - phased_stdevs) + [cycle[0][var_idx]],
                             list(phased_means + phased_stdevs) + [cycle[0][var_idx]],
                             color='grey',
                             alpha=0.25)
            # dashed cycle in `converging` region
            plt.plot(range(len(phased_means) - len(cycle), len(phased_means) + 1),
                     [cycle[idx][var_idx] for idx in range(len(cycle))] + [cycle[0][var_idx]],
                     color='orange', linestyle='dashed')
            # solid cycle in `converged` region
            plt.plot(range(len(phased_means), len(phased_means) + len(cycle)),
                     [cycle[idx][var_idx] for idx in range(len(cycle))],
                     color='orange')

            plt.title(f'Phased Envelope of trajectories for {var}')

            plt.xlabel('Distance to phase-fixing point on limit cycle')
            plt.xticks(range(len(phased_means) + len(cycle) + 1),
                       list(range(len(phased_means), -1 - len(cycle), -1)),
                       rotation=45)
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
            length_dist = np.bincount(trajectory_lengths[limit_set])
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

    performed_complete_search = False  # TODO: redo, maybe not here

    cycles = [{'states': cycle,
               'len': len(cycle),
               'count': counts[frozenset(cycle)],
               'percent': 100 * counts[frozenset(cycle)] / num_iterations,
               'len-dist-image': length_distribution_images[frozenset(cycle)] if frozenset(
                   cycle) in length_distribution_images else "",
               'limit-set-stats-images': limit_set_stats_images[frozenset(cycle)] if frozenset(
                   cycle) in limit_set_stats_images else dict(), }
              for cycle in cycle_list]
    # respond with the results-of-computation page
    response = make_response(render_template('compute-cycles.html',
                                             cycles=cycles,
                                             variables=variables,
                                             cycle_len_counts=cycle_len_counts,
                                             complete_results=performed_complete_search))
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
