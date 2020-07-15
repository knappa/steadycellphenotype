import os
import shutil
import subprocess
import tempfile

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
from flask import Markup, make_response
from equation_system import EquationSystem
import networkx as nx

from ._util import *


def run_model_with_init_val(init_state, knockout_model, variables, continuous, model_state, equation_system):
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
                init_state_params += [key, value]

        convert_to_c_process = \
            subprocess.run([os.getcwd() + '/convert.py', '-graph',
                            '--count', '1',
                            '-i', tmp_dir_name + '/model.txt',
                            '-o', tmp_dir_name + '/model.c'] + init_state_params + continuity_params,
                           capture_output=True)

        if convert_to_c_process.returncode != 0:
            response = make_response(error_report(
                'Error running converter!\n{}\n{}'.format(html_encode(convert_to_c_process.stdout),
                                                          html_encode(convert_to_c_process.stderr))))
            return response_set_model_cookie(response, model_state)

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
            response = make_response(error_report(
                'Error running compiler!\n{}\n{}'.format(html_encode(compilation_process.stdout),
                                                         html_encode(compilation_process.stderr))))
            return response_set_model_cookie(response, model_state)

        simulation_process = \
            subprocess.run([tmp_dir_name + '/model'], capture_output=True)
        if simulation_process.returncode != 0:
            response = make_response(error_report(
                'Error running simulator!\n{}\n{}'.format(html_encode(simulation_process.stdout),
                                                          html_encode(simulation_process.stderr))))
            return response_set_model_cookie(response, model_state)

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


def run_model_variable_init_vals(init_state, knockout_model, variables, continuous, model_state, equation_system):
    """ Run the model on possibly *'ed sets of initial conditions """
    # find any *'ed variables
    variable_states = [k for k in init_state if init_state[k] == '*']

    if len(variable_states) == 0:
        # if there aren't any, just run the model
        return [
            run_model_with_init_val(init_state, knockout_model, variables, continuous, model_state, equation_system)
            ]
    else:
        # in there are, get one and try all possible values. Recurse.
        variable_state = variable_states[0]
        edge_lists = []
        for val in range(3):
            var_init_state = init_state.copy()
            var_init_state[variable_state] = str(val)
            edge_lists += run_model_variable_init_vals(var_init_state,
                                                       knockout_model,
                                                       variables,
                                                       continuous,
                                                       model_state,
                                                       equation_system)
        return edge_lists


def compute_trace(model_state, knockout_model, variables, continuous, init_state):
    """ Run the cycle finding simulation for an initial state """
    # TODO: initially copied from compute_cycles, should look for code duplication and refactoring
    #  opportunities
    equation_system = EquationSystem(model_state['model'])

    edge_lists = run_model_variable_init_vals(init_state,
                                              knockout_model,
                                              variables,
                                              continuous,
                                              model_state,
                                              equation_system)

    def to_key(edge):
        return frozenset(edge.items())

    # give numeric labels to vertices
    labels = dict()
    count = 0
    for edge_list in edge_lists:
        for edge in edge_list:
            source = to_key(edge['source'])
            if source not in labels:
                labels[source] = count
                count += 1

    return_states = []
    for edge_list in edge_lists:
        return_state = to_key(edge_list[-1]['target'])
        return_states.append(labels[return_state])

    source_labels = [[labels[to_key(edge['source'])] for edge in edge_list] for edge_list in edge_lists]

    # trace visualization
    g = nx.DiGraph()
    for edge_list in edge_lists:
        for edge in edge_list:
            source = to_key(edge['source'])
            target = to_key(edge['target'])
            g.add_edge(labels[source], labels[target])

    # draw the damned thing
    plt.rcParams['svg.fonttype'] = 'none'
    plt.figure(figsize=(4, 3))
    #nx.draw_kamada_kawai(g, connectionstyle='arc3,rad=0.2', with_labels=True)
    nx.draw_spectral(g,
                     #connectionstyle='arc3,rad=0.2',
                     with_labels=True)
    plt.title('Trajectory')
    image_filename = 'trajectory.svg'
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
        plt.close()
        with open(tmp_dir_name + '/' + image_filename, 'r') as image:
            trajectory_image = Markup(image.read())

    # respond with the results-of-computation page
    response = make_response(render_template('compute-trace.html',
                                             variables=equation_system.target_variables(),
                                             num_edge_lists=len(edge_lists),
                                             trajectories=zip(edge_lists,
                                                              return_states,
                                                              source_labels,
                                                              map(len, edge_lists)),
                                             trajectory_image=trajectory_image))
    return response_set_model_cookie(response, model_state)
