import os
import shutil
import subprocess
import tempfile

import matplotlib
import matplotlib.pyplot as plt
from flask import Markup, make_response

from ._util import *

matplotlib.use('agg')


def compute_cycles(model_state, knockout_model, variables, continuous, num_iterations):
    """ Run the cycle finding simulation """
    # TODO: better integrating, thinking more about security
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

        # randomized search is kind of silly if you ask for more iterations than there are actual states
        # so go to the complete search mode in this case.
        if 3 ** len(variables) <= num_iterations:
            complete_search_params = ['-complete_search']
        else:
            complete_search_params = []

        convert_to_c_process = \
            subprocess.run([os.getcwd() + '/convert.py', '-sim',
                            '--count', str(num_iterations),
                            '-i', tmp_dir_name + '/model.txt',
                            '-o', tmp_dir_name + '/model.c'] + continuity_params + complete_search_params,
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
        # not needed (except for graph simulator)
        # subprocess.run(['cp', os.getcwd() + '/link-table.h', tmp_dir_name])

        # be fancy about compiler selection
        installed_compilers = [shutil.which('clang'), shutil.which('gcc'), shutil.which('cc')]
        compiler = installed_compilers[0] if installed_compilers[0] is not None \
            else installed_compilers[1] if installed_compilers[1] is not None \
            else installed_compilers[2]

        compilation_process = \
            subprocess.run([compiler, '-O3', tmp_dir_name + '/model.c', '-o', tmp_dir_name + '/model'],
                           capture_output=True)
        if compilation_process.returncode != 0:
            with open(tmp_dir_name + '/model.c', 'r') as source_file:
                response = make_response(error_report(
                    'Error running compiler!\n{}\n{}\n{}'.format(html_encode(compilation_process.stdout),
                                                                 html_encode(compilation_process.stderr),
                                                                 html_encode(source_file.read()))))
            return response_set_model_cookie(response, model_state)

        simulation_process = \
            subprocess.run([tmp_dir_name + '/model'], capture_output=True)
        if simulation_process.returncode != 0:
            response = make_response(error_report(
                'Error running simulator!\n{}\n{}'.format(html_encode(simulation_process.stdout),
                                                          html_encode(simulation_process.stderr))))
            return response_set_model_cookie(response, model_state)

        try:
            simulator_output = json.loads(simulation_process.stdout.decode())
        except json.decoder.JSONDecodeError:
            response = make_response(error_report(
                'Error decoding simulator output!\n{}\n{}'.format(html_encode(simulation_process.stdout),
                                                                  html_encode(simulation_process.stderr))))
            return response_set_model_cookie(response, model_state)

        # somewhat redundant data in the two fields, combine them, indexed by id
        combined_output = {
            cycle['id']: {'length': cycle['length'],
                          'count': cycle['count'],
                          'percent': cycle['percent'],
                          'length-dist': cycle['length-dist']}
            for cycle in simulator_output['counts']}
        for cycle in simulator_output['cycles']:
            combined_output[cycle['id']]['cycle'] = cycle['cycle']

        cycle_list = list(combined_output.values())
        cycle_list.sort(key=lambda cycle: cycle['count'], reverse=True)

        from collections import defaultdict
        cycle_len_counts = defaultdict(lambda: 0)
        for cycle in cycle_list:
            cycle_len_counts[cycle['length']] += 1
        cycle_len_counts = tuple(sorted(cycle_len_counts.items()))

        # create length distribution plots
        plt.rcParams['svg.fonttype'] = 'none'
        for cycle in combined_output:
            length_dist = combined_output[cycle]['length-dist']
            plt.figure(figsize=(4, 3))
            plt.bar(x=range(len(length_dist)),
                    height=length_dist,
                    color='#002868')
            plt.title('Distribution of lengths of paths')
            plt.xlabel('Length of path')
            plt.ylabel('Number of states')
            plt.tight_layout()
            image_filename = 'dist-' + str(cycle) + '.svg'
            plt.savefig(tmp_dir_name + '/' + image_filename, transparent=True, pad_inches=0.0)
            plt.close()
            with open(tmp_dir_name + '/' + image_filename, 'r') as image:
                combined_output[cycle]['image'] = Markup(image.read())

    performed_complete_search = simulator_output['complete_search'] if 'complete_search' in simulator_output else False

    # respond with the results-of-computation page
    response = make_response(render_template('compute-cycles.html',
                                             cycles=cycle_list,
                                             cycle_len_counts=cycle_len_counts,
                                             complete_results=performed_complete_search))
    return response_set_model_cookie(response, model_state)
