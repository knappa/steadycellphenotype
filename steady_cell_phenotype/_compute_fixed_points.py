import shutil
import subprocess
import tempfile

from flask import make_response

from steady_cell_phenotype._util import *


def compute_fixed_points(knockout_model, variables, continuous):
    """ Run the fixed-point finding computation """

    # check to make sure that we have macaulay installed
    macaulay_executable = shutil.which('M2')
    if macaulay_executable is None:
        return make_response(
                error_report("Macaulay2 was not found on the server; we cannot do as you ask."))

    # we are set to do the computation
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        with open(tmp_dir_name + '/model.txt', 'w') as model_file:
            model_file.write(knockout_model)
            model_file.write('\n')

        # convert the model to polynomials
        non_continuous_vars = [variable for variable in variables if not continuous[variable]]
        if len(non_continuous_vars) > 0:
            continuity_params = ['-c', '-comit'] + [variable for variable in variables if
                                                    not continuous[variable]]
        else:
            continuity_params = ['-c']

        convert_to_poly_process = \
            subprocess.run([get_resource_path('scp_converter.py'), '-n',
                            '-i', tmp_dir_name + '/model.txt',
                            '-o', tmp_dir_name + '/model-polys.txt'] + continuity_params,
                           capture_output=True)

        if convert_to_poly_process.returncode != 0:
            return make_response(error_report(
                    'Error running converter!\n{}\n{}'.format(
                            html_encode(convert_to_poly_process.stdout),
                            html_encode(convert_to_poly_process.stderr))))

        template_contents = get_text_resource('find_steady_states.m2-template')
        with open(tmp_dir_name + '/find_steady_states.m2', 'w') as macaulay_script:
            macaulay_script.write(template_contents.format(
                    polynomial_file=tmp_dir_name + '/model-polys.txt',
                    output_file=tmp_dir_name + '/results.txt'))

        # TODO: put a limit on the amount of time this can run
        find_steady_states_process = \
            subprocess.run([macaulay_executable,
                            '--script',
                            tmp_dir_name + '/find_steady_states.m2'],
                           capture_output=True)
        if find_steady_states_process.returncode != 0:
            # this string appears when there are no solutions. I don't think it appears
            # otherwise.
            if '-infinity (of class InfiniteNumber)' in str(find_steady_states_process.stderr):
                return make_response(message(
                        'Macaulay could not find any fixed points, '
                        'this is likely because there aren\'t any.'))
            else:
                with open(tmp_dir_name + '/model-polys.txt', 'r') as poly_file:
                    return make_response(error_report(
                            'Error running Macaulay!\n<br>\n{}\n<br>\n{}\n<br>\n{}'.format(
                                    html_encode(find_steady_states_process.stdout),
                                    html_encode(find_steady_states_process.stderr),
                                    html_encode(poly_file.read()))))

        def process_line(line):
            line = line.strip()
            if len(line) < 2:  # should at least have {}
                return None
            elif line[0] != '{' or line[-1] != '}':
                raise RuntimeError("Malformed response from Macaulay")

            line = line[1:-1]  # strip the {}'s
            try:
                results = map(int, line.split(','))
            except ValueError:
                raise RuntimeError("Malformed response from Macaulay")

            # make sure they are in 0,1,2
            results = map(lambda n: (n + 3) % 3, results)

            return tuple(results)

        try:
            with open(tmp_dir_name + '/results.txt', 'r') as file:
                fixed_points = [process_line(line) for line in file]
        except (IOError, RuntimeError):
            return make_response(error_report('Malformed response from Macaulay!'))

    # respond with the results-of-computation page
    return make_response(
            render_template('compute-fixed-points.html', variables=variables,
                            fixed_points=fixed_points))
