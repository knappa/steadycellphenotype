import os
import shutil
import subprocess
import tempfile

from flask import make_response, Markup, Response
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from equation_system import EquationSystem
from ._util import *

matplotlib.use('agg')


def run_model_with_init_val(init_state, knockout_model, variables, continuous, equation_system):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        with open(tmp_dir_name + '/model.txt', 'w') as model_file:
            model_file.write(knockout_model)
            model_file.write('\n')

        non_continuous_vars = [variable for variable in variables if not continuous[variable]]
        if len(non_continuous_vars) > 0:
            continuity_params = ['-c', '-comit'] + [variable for variable in variables
                                                    if not continuous[variable]]
        else:
            continuity_params = ['-c']

        init_state_params = []
        if len(init_state) > 0:
            init_state_params += ['-init-val']
            for key, value in init_state.items():
                init_state_params += [key, str(value)]

        convert_to_c_process = \
            subprocess.run([os.getcwd() + '/convert.py', '-graph',
                            '--count', '1',
                            '-i', tmp_dir_name + '/model.txt',
                            '-o', tmp_dir_name + '/model.c'] + init_state_params + continuity_params,
                           capture_output=True)

        if convert_to_c_process.returncode != 0:
            return make_response(error_report(
                'Error running converter!\n{}\n{}'.format(html_encode(convert_to_c_process.stdout),
                                                          html_encode(convert_to_c_process.stderr))))

        # copy the header files over
        subprocess.run(['cp', os.getcwd() + '/mod3ops.h', tmp_dir_name])
        subprocess.run(['cp', os.getcwd() + '/bloom-filter.h', tmp_dir_name])
        subprocess.run(['cp', os.getcwd() + '/cycle-table.h', tmp_dir_name])
        subprocess.run(['cp', os.getcwd() + '/length-count-array.h', tmp_dir_name])
        subprocess.run(['cp', os.getcwd() + '/link-table.h', tmp_dir_name])

        # be fancy about compiler selection
        installed_compilers = [shutil.which('clang'), shutil.which('gcc'), shutil.which('cc')]
        compiler = installed_compilers[0] if installed_compilers[0] is not None \
            else installed_compilers[1] if installed_compilers[1] is not None \
            else installed_compilers[2]

        compilation_process = \
            subprocess.run([compiler, '-O3', tmp_dir_name + '/model.c', '-o', tmp_dir_name + '/model'],
                           capture_output=True)
        if compilation_process.returncode != 0:
            return make_response(error_report(
                'Error running compiler!\n{}\n{}'.format(html_encode(compilation_process.stdout),
                                                         html_encode(compilation_process.stderr))))

        simulation_process = \
            subprocess.run([tmp_dir_name + '/model'], capture_output=True)
        if simulation_process.returncode != 0:
            return make_response(error_report(
                'Error running simulator!\n{}\n{}'.format(html_encode(simulation_process.stdout),
                                                          html_encode(simulation_process.stderr))))

        simulator_output = json.loads(simulation_process.stdout.decode())
        edge_list = simulator_output['edges']

        num_variables = len(equation_system.target_variables())
        edge_list = [{'source': dict(zip(equation_system.target_variables(),
                                         decode_int(edge['source'], num_variables))),
                      'target': dict(zip(equation_system.target_variables(),
                                         decode_int(edge['target'], num_variables))),
                      'step': int(edge['step'])}
                     for edge in edge_list]
        edge_list = sorted(edge_list, key=lambda edge: edge['step'])
    return edge_list


def run_model_variable_initial_values(init_state,
                                      knockout_model,
                                      variables,
                                      continuous,
                                      equation_system,
                                      check_nearby):
    """
    Run the model on possibly *'ed sets of initial conditions

    Parameters
    ----------
    init_state
    knockout_model
    variables
    continuous
    equation_system
    check_nearby

    Returns
    -------
    List
    """

    # deal with any *'ed variables
    def get_states(initial_state, remaining_variable_states):
        if len(remaining_variable_states) == 0:
            return [initial_state]
        else:
            initial_states = []
            for val in range(3):
                resolved_state = initial_state.copy()
                resolved_state[remaining_variable_states[0]] = val
                initial_states += get_states(resolved_state, remaining_variable_states[1:])
            return initial_states

    variable_states = [k for k in init_state if init_state[k] == '*']
    init_states = get_states(init_state, variable_states)

    # check to see if we will be overloaded
    if len(variable_states) > MAX_SUPPORTED_VARIABLE_STATES:
        return make_response(error_report(
            f"Web platform is limited to {MAX_SUPPORTED_VARIABLE_STATES} variable states."))

    if check_nearby:
        nearby_states = []
        for state in init_states:
            if state not in nearby_states:
                nearby_states.append(state)
            for variable in variables:
                if str(state[variable]) == '0':
                    near_state = state.copy()
                    near_state[variable] = '1'
                    if near_state not in nearby_states:
                        nearby_states.append(near_state)
                elif str(state[variable]) == '1':
                    near_state = state.copy()
                    near_state[variable] = '0'
                    if near_state not in nearby_states:
                        nearby_states.append(near_state)
                    near_state = state.copy()
                    near_state[variable] = '2'
                    if near_state not in nearby_states:
                        nearby_states.append(near_state)
                elif str(state[variable]) == '2':
                    near_state = state.copy()
                    near_state[variable] = '1'
                    if near_state not in nearby_states:
                        nearby_states.append(near_state)
                else:
                    assert False, "invalid state!: '" + str(state[variable]) + "' " + str(type(state[variable]))
        init_states = nearby_states

    edge_lists = []
    for state in init_states:
        edge_lists.append(run_model_with_init_val(state,
                                                  knockout_model,
                                                  variables,
                                                  continuous,
                                                  equation_system))
    return edge_lists


def connected_component_layout(g: nx.DiGraph):
    """
    lay out a graph with a single connected component,
    returns dictionary of positions and width/height of bounding box
    """

    # get attractor (fixed point or cycle)
    attractor_set = next(nx.attracting_components(g))
    cycle_len = len(attractor_set)

    # no guarantee the attractor set is in the proper order:
    base_point = next(iter(attractor_set))
    cycle = [base_point]
    # in python 3.8+ you have assignment expressions:
    # while (next_point := list(g.successors(cycle[-1]))[0]) != base_point:
    #    cycle.append(next_point)
    next_point = list(g.successors(cycle[-1]))[0]
    while next_point != base_point:
        cycle.append(next_point)
        next_point = list(g.successors(cycle[-1]))[0]

    pos = dict()

    # Note: networkx has 'node_size'==300 but it is unclear what units those are
    def recurse_layout(successor, level, max_theta, min_theta):
        predecessors = [predecessor
                        for predecessor in g.predecessors(successor)
                        if predecessor != successor and predecessor not in pos]
        if len(predecessors) == 0:
            return
        delta_theta = (max_theta - min_theta) / (len(predecessors) + 1)
        for k, predecessor in enumerate(predecessors):
            theta_k = min_theta + (k + 1) * delta_theta
            pos[predecessor] = level * np.array([np.cos(theta_k),
                                                 np.sin(theta_k)])
            recurse_layout(predecessor, level + 1, min_theta + (k + 1.5) * delta_theta,
                           min_theta + (k + 0.5) * delta_theta)

    # lay out the cycle:
    if cycle_len == 1:
        pos[base_point] = np.array([0.0, 0.0])
        recurse_layout(base_point, 1, 2 * np.pi, 0)
    else:
        for n, point in enumerate(cycle):
            theta = 2 * np.pi * (n + 0.5) / cycle_len
            pos[point] = np.array([np.cos(theta), np.sin(theta)])
        for n, point in enumerate(cycle):
            recurse_layout(point, 2,
                           2 * np.pi * (n + 1) / cycle_len,
                           2 * np.pi * n / cycle_len)
    # move corner
    pos_array = np.array(list(pos.values()))
    offset = np.min(pos_array, axis=0)
    pos = {node: pt - offset for node, pt in pos.items()}
    return pos, np.max(pos_array, axis=0) - offset


def graph_layout(g):
    # lay out connected components, in bounding boxes. then offset
    # noinspection PyTypeChecker
    components_layouts = [
        connected_component_layout(nx.subgraph_view(g, filter_node=lambda vertex: vertex in component_vertices))
        for component_vertices in nx.weakly_connected_components(g)]
    pos = dict()
    corner = np.array([0.0, 0.0])
    running_y = 0.0
    for component_pos, geom in components_layouts:
        running_y = max(running_y, geom[1])
        for node in component_pos:
            pos[node] = component_pos[node] + corner
        corner += np.array([geom[0] + 1.0, 0])
        if corner[0] > 20.0:
            corner[0] = 0
            corner[1] += running_y
            running_y = 0.0
    return pos


def compute_trace(model_text, knockout_model, variables, continuous, init_state, check_nearby):
    """ Run the cycle finding simulation for an initial state """
    # TODO: initially copied from compute_cycles, should look for code duplication and refactoring
    #  opportunities
    equation_system = EquationSystem.from_text(model_text)

    edge_lists = run_model_variable_initial_values(init_state,
                                                   knockout_model,
                                                   variables,
                                                   continuous,
                                                   equation_system,
                                                   check_nearby)

    # can return responses, if there is an error, return any such error response
    for edge in edge_lists:
        if isinstance(edge, Response):
            return edge

    def to_key(edges):
        return frozenset(edges.items())

    # give numeric labels to vertices
    labels = dict()
    count = 0
    for edge_list in edge_lists:
        for edge in edge_list:
            source = to_key(edge['source'])
            if source not in labels:
                labels[source] = count
                count += 1
            target = to_key(edge['target'])
            if target not in labels:
                labels[target] = count
                count += 1

    return_states = []
    for edge_list in edge_lists:
        return_state = to_key(edge_list[-1]['target'])
        return_states.append(labels[return_state])

    source_labels = [[labels[to_key(edge['source'])] for edge in edge_list] for edge_list in edge_lists]

    # draw a visualization of variable levels
    variable_level_plots = []
    for n, edge_list in enumerate(edge_lists):
        plt.rcParams['svg.fonttype'] = 'none'
        plt.figure(figsize=(6, 2))

        # prepare to plot
        data = np.array([
            [edge['source'][variable] for edge in edge_list] + [edge_list[-1]['target'][variable]]
            for variable in variables]).T

        # plot
        plt.plot(data)

        # note the repeat region
        if return_states[n] in source_labels[n]:
            plt.axvline(source_labels[n].index(return_states[n]), color='k', linestyle='dotted')
        plt.axvline(len(edge_list), color='k', linestyle='dotted')

        # limits and ticks
        plt.ylim([-0.25, 2.25])
        plt.yticks([0, 1, 2])
        plt.xlim([-0.25, len(edge_list) + 0.25])
        plt.xticks(np.arange(len(edge_list) + 1))
        plt.gca().set_xticklabels(source_labels[n] + [return_states[n]])

        # legend and labels
        plt.legend(variables, bbox_to_anchor=(1.04, 1), loc="center left")
        plt.ylabel("Level")
        plt.xlabel("State")
        plt.tight_layout()
        image_filename = f'levels{n}.svg'
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_dir_name + '/' + image_filename, 'r') as image:
                variable_level_plots.append(Markup(image.read()))

    # trace visualization
    g = nx.DiGraph()
    for edge_list in edge_lists:
        for edge in edge_list:
            source = to_key(edge['source'])
            target = to_key(edge['target'])
            g.add_edge(labels[source], labels[target])

    # lay out the graph
    pos = graph_layout(g)

    # get overall geometry
    pos_array = np.array(list(pos.values()))
    width, height = np.max(pos_array, axis=0) - np.min(pos_array, axis=0)

    # draw the damned thing
    plt.rcParams['svg.fonttype'] = 'none'
    fig_height = min(3, max(100, width / height))
    plt.figure(figsize=(4, fig_height))
    nx.draw(g,
            pos=pos,
            with_labels=True)
    plt.title('Trajectory')
    image_filename = 'trajectory.svg'
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
        plt.close()
        with open(tmp_dir_name + '/' + image_filename, 'r') as image:
            trajectory_image = Markup(image.read())

    # respond with the results-of-computation page
    return make_response(render_template('compute-trace.html',
                                         variables=equation_system.target_variables(),
                                         num_edge_lists=len(edge_lists),
                                         trajectories=list(zip(edge_lists,
                                                               return_states,
                                                               source_labels,
                                                               map(len, edge_lists),
                                                               variable_level_plots)),
                                         trajectory_image=trajectory_image))
