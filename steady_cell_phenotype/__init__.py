#!/usr/bin/env python3
#
# PROGRAM DESCRIPTION TODO
#
# Copyright Adam C. Knapp 2019
# License: CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/
# Funded by American University Mellon Grant

import datetime
import json
import os
import string
import subprocess
import tempfile

from flask import Flask, render_template, request, make_response, Response, Markup
from werkzeug.utils import secure_filename

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY='dev',
        MAX_CONTENT_LENGTH = 512 * 1024, # maximum upload size: 512 kilobytes
        UPLOAD_FOLDER=tempfile.TemporaryDirectory(), # TODO: does this make sense??? doesn't work without it
        # added to assist when developing, should be removed in production
        TEMPLATES_AUTO_RELOAD=True,
        )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
        
    ####################################################################################################
    # some _effectively_ static pages, which share a basic template

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('page_not_found.html'), 404

    @app.route('/about/')
    def about():
        """ render the about page """
        return render_template('about.html')

    @app.route('/quickstart/')
    def quickstart():
        """ render the quick start page """
        return render_template('quickstart.html')

    @app.route('/docs/')
    def docs():
        """ render the documentation page """
        return render_template('docs.html')

    @app.route('/source/')
    def source():
        """ render the links-to-source page """
        return render_template('source.html')

    ####################################################################################################
    # a download page

    @app.route('/download-tsv/', methods=['POST'])
    def download_tsv():
        try:
            tsv = request.form['model_result'].strip()
            return Response(
                tsv,
                mimetype="text/tab-separated-values",
                headers={"Content-disposition":
                             "attachment; filename=model_result.tsv"})
        except:
            response = make_response(error_report(
                "Something odd happened."))
            return response

    ####################################################################################################
    # the main / model entry page

    @app.route('/', methods=['GET','POST'])
    def index():
        """ render the main page """

        # get model file, if present
        model_from_file = None
        if request.method == 'POST' and 'model-file' in request.files:
            model_file = request.files['model-file']
            if model_file.filename != '':
                with tempfile.TemporaryDirectory() as tmpdirname:
                    filename = tmpdirname + '/' + secure_filename(model_file.filename)
                    model_file.save(filename)
                    with open(filename, 'r') as file:
                        model_from_file = file.read().strip()
        
        model_state_cookie = request.cookies.get('state')
        if model_state_cookie is not None and model_state_cookie != '':
            model_state = json.loads(model_state_cookie)
            model_text = model_state['model']
            if model_from_file is not None:
                model_text = model_from_file
            model_lines = model_text.count('\n')
            return render_template('index.html', model_text=model_text, rows=max(10, model_lines + 2))
        else:
            return render_template('index.html', rows=10)

    ####################################################################################################
    # the main computational page

    @app.route('/compute/', methods=['POST'])
    def compute():
        """ render the results of computation page """

        # load existing state cookie, if it exists and makes sense
        model_state_cookie = request.cookies.get('state')
        if model_state_cookie is not None and model_state_cookie != '':
            try:
                model_state = json.loads(model_state_cookie)
            except:
                response = make_response(error_report(
                    "For some reason, we could not parse the cookie for this site. " +
                    "We just tried to clear it, but if the error persists clear the cookie and try again."))
                return response_set_model_cookie(response, dict())
        else:
            # respond with an error message if submission is ill-formed
            response = make_response(error_report(
                'We couldn\'t find the cookie which contains your model, please go back to the main page and try again'))
            return response_set_model_cookie(response, dict())

        # respond with an error message if submission is ill-formed
        if 'model' not in model_state:
            response = make_response(error_report(
                'The cookie which contains your model is ill-formed, please go back to the main page and try again'))
            return response_set_model_cookie(response, dict())

        # get the variable list and right hand sides
        model = model_state['model']
        try:
            # attempt to get the variable list
            variables, right_sides = get_model_variables(model)
        except Exception as e:
            # respond with an error message if submission is ill-formed
            response = make_response(error_report(str(e)))
            return response_set_model_cookie(response, model_state)

        # decide which variables the user specified as continuous
        continuous = {variable.strip(): True if '{}-continuous'.format(variable) in request.form else False
                      for variable in variables}

        # create knockout model i.e. set certain values to constants without deleting the formulae
        knockouts = {variable.strip(): request.form['{}-KO'.format(variable)]
                     for variable in variables
                     if request.form['{}-KO'.format(variable)] != 'None'}

        knockout_model = ""
        for variable, rhs in zip(variables, right_sides):
            if variable in knockouts:
                knockout_model += "{v} = {r}\n".format(v=variable, r=knockouts[variable])
            else:
                knockout_model += "{v} = {r}\n".format(v=variable, r=rhs)

        # get number of iterations for the simulation
        try:
            num_iterations = int(request.form['num_samples'])
        except:
            num_iterations = 0  # not going to waste any effort on garbage

        # decide which type of computation to run
        if 'action' not in request.form or request.form['action'] not in ['cycles', 'fixed_points']:
            response = make_response(error_report(
                'The request was ill-formed, please go back to the main page and try again'))
            return response_set_model_cookie(response, model_state)
        elif request.form['action'] == 'cycles':
            return compute_cycles(model_state, knockout_model, variables, continuous, num_iterations)
        else:  # request.form['action'] == 'fixed_points'
            return compute_fixed_points(model_state, knockout_model, variables, continuous)

    def compute_fixed_points(model_state, knockout_model, variables, continuous):
        """ Run the fixed-point finding computation """

        # check to make sure that we have macaulay installed
        import shutil
        macaulay_executable = shutil.which('M2')
        if macaulay_executable is None:
            response = make_response(error_report("Macaulay2 was not found on the server; we cannot do as you ask."))
            return response_set_model_cookie(response, model_state)

        # we are set to do the computation
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + '/model.txt', 'w') as model_file:
                model_file.write(knockout_model)
                model_file.write('\n')

            def error_msg_parse(msg):
                import html
                if type(msg) is not str:
                    msg = msg.decode()
                return html.escape(msg).replace('\n', '<br>').replace(' ', '&nbsp')

            # convert the model to polynomials
            non_continuous_vars = [variable for variable in variables if not continuous[variable]]
            if len(non_continuous_vars) > 0:
                continuity_params = ['-c', '-comit'] + [variable for variable in variables if not continuous[variable]]
            else:
                continuity_params = ['-c']

            convert_to_poly_process = \
                subprocess.run([os.getcwd() + '/convert.py',  # '-n',
                                '-i', tmpdirname + '/model.txt',
                                '-o', tmpdirname + '/model-polys.txt'] + continuity_params,
                               capture_output=True)
            ### XXX Jan 8
            print(convert_to_poly_process.args)
            print(convert_to_poly_process.stdout)
            #import shutil
            #shutil.copyfile(tmpdirname + '/model-polys.txt', '/home/knappa/model-polys.txt')
            ### XXX Jan 8
            if convert_to_poly_process.returncode != 0:
                response = make_response(error_report(
                    'Error running converter!\n{}\n{}'.format(error_msg_parse(convert_to_poly_process.stdout),
                                                              error_msg_parse(convert_to_poly_process.stderr))))
                return response_set_model_cookie(response, model_state)

            with open(os.getcwd() + '/find_steady_states.m2-template', 'r') as template, \
                    open(tmpdirname + '/find_steady_states.m2', 'w') as macaulay_script:
                template_contents = template.read()
                macaulay_script.write(template_contents.format(
                    polynomial_file=tmpdirname + '/model-polys.txt',
                    output_file=tmpdirname + '/results.txt'))
                
            # TODO: put a limit on the amount of time this can run
            find_steady_states_process = \
                subprocess.run([macaulay_executable,
                                '--script',
                                tmpdirname + '/find_steady_states.m2'],
                               capture_output=True)
            if find_steady_states_process.returncode != 0:
                with open(tmpdirname + '/model-polys.txt','r') as poly_file:
                    response = make_response(error_report(
                        'Error running Macaulay!\n<br>\n{}\n<br>\n{}\n<br>\n{}'.format(
                            error_msg_parse(find_steady_states_process.stdout),
                            error_msg_parse(find_steady_states_process.stderr),
                            error_msg_parse(poly_file.read()))))
                return response_set_model_cookie(response, model_state)

            def process_line(line):
                line = line.strip()
                if len(line) < 2:  # should at least have {}
                    return None
                elif '{' is not line[0] or '}' is not line[-1]:
                    raise Error("Malformed response from Macaulay")

                line = line[1:-1]  # strip the {}'s
                try:
                    results = map(int, line.split(','))
                except:
                    raise Error("Malformed response from Macaulay")

                # make sure they are in 0,1,2
                results = map(lambda n: (n + 3) % 3, results)

                return tuple(results)

            try:
                with open(tmpdirname + '/results.txt', 'r') as file:
                    fixed_points = [process_line(line) for line in file]
            except:
                response = make_response(error_report('Malformed response from Macaulay!'))
                return response_set_model_cookie(response, model_state)

        # respond with the results-of-computation page
        response = make_response(
            render_template('compute-fixed-points.html', variables=variables, fixed_points=fixed_points))
        return response_set_model_cookie(response, model_state)

    def compute_cycles(model_state, knockout_model, variables, continuous, num_iterations):
        """ Run the cycle finding simulation """
        # Oh, this seems so very ugly
        # TODO: better integrating, thinking more about security
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + '/model.txt', 'w') as model_file:
                model_file.write(knockout_model)
                model_file.write('\n')

            def error_msg_parse(msg):
                import html
                if type(msg) is not str:
                    msg = msg.decode()
                return html.escape(msg).replace('\n', '<br>').replace(' ', '&nbsp')

            non_continuous_vars = [variable for variable in variables if not continuous[variable]]
            if len(non_continuous_vars) > 0:
                continuity_params = ['-c', '-comit'] + [variable for variable in variables if not continuous[variable]]
            else:
                continuity_params = ['-c']
            convert_to_c_process = \
                subprocess.run([os.getcwd() + '/convert.py', '-n', '-sim',
                                '--count', str(num_iterations),
                                # '10000',  # only 10K runs, I don't want to overload the server
                                '-i', tmpdirname + '/model.txt',
                                '-o', tmpdirname + '/model.c'] + continuity_params,
                               capture_output=True)
            if convert_to_c_process.returncode != 0:
                response = make_response(error_report(
                    'Error running converter!\n{}\n{}'.format(error_msg_parse(convert_to_c_process.stdout),
                                                              error_msg_parse(convert_to_c_process.stderr))))
                return response_set_model_cookie(response, model_state)

            # copy the header files over
            subprocess.run(['cp', os.getcwd() + '/mod3ops.h', tmpdirname])
            subprocess.run(['cp', os.getcwd() + '/bloom-filter.h', tmpdirname])
            subprocess.run(['cp', os.getcwd() + '/cycle-table.h', tmpdirname])
            subprocess.run(['cp', os.getcwd() + '/length-count-array.h', tmpdirname])

            # TODO: be fancy about compiler selection, using shutil.which
            compilation_process = \
                subprocess.run(['gcc', '-O3', tmpdirname + '/model.c', '-o', tmpdirname + '/model'],
                               capture_output=True)
            if compilation_process.returncode != 0:
                response = make_response(error_report(
                    'Error running compiler!\n{}\n{}'.format(compilation_process.stdout.decode(),
                                                             compilation_process.stderr.decode())))
                return response_set_model_cookie(response, model_state)

            simulation_process = \
                subprocess.run([tmpdirname + '/model'], capture_output=True)
            if simulation_process.returncode != 0:
                response = make_response(error_report(
                    'Error running simulator!\n{}\n{}'.format(simulation_process.stdout.decode(),
                                                              simulation_process.stderr.decode())))
                return response_set_model_cookie(response, model_state)

            simulator_output = json.loads(simulation_process.stdout.decode())

            # somewhat redundant data in the two fields, combine them, indexed by id
            combined_output = {
                cycle['id']: {'length': cycle['length'],
                            'count': cycle['count'],
                            'percent': cycle['percent'],
                            'length-dist': cycle['length-dist'] }
                for cycle in simulator_output['counts'] }
            for cycle in simulator_output['cycles']:
                combined_output[cycle['id']]['cycle'] = cycle['cycle']

            cycle_list = list(combined_output.values())
            cycle_list.sort(key=lambda cycle: cycle['count'], reverse=True)

            # create length distribution plots
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            plt.rcParams['svg.fonttype'] = 'none'
            for cycle in combined_output:
                length_dist = combined_output[cycle]['length-dist']
                plt.figure(figsize=(4,3))
                plt.bar(x=range(len(length_dist)), height=length_dist, color='#002868')
                plt.title('Distribution of lengths of paths')
                plt.xlabel('Length of path')
                plt.ylabel('Number of states')
                plt.tight_layout()
                image_filename = 'dist-' + str(cycle) + '.svg'
                plt.savefig(tmpdirname + '/' + image_filename, transparent=True, pad_inches=0.0)
                with open(tmpdirname + '/' + image_filename, 'r') as image:
                    combined_output[cycle]['image'] = Markup(image.read())

        # respond with the results-of-computation page
        response = make_response(render_template('compute-cycles.html', cycles=cycle_list))
        return response_set_model_cookie(response, model_state)

    ####################################################################################################
    # model options page

    @app.route('/options/', methods=['POST'])
    def options():
        """ model options page """
        # get submitted model from form
        model = request.form['model'].strip()

        # load existing state cookie, if it exists, and update model
        model_state_cookie = request.cookies.get('state')
        if model_state_cookie is not None and model_state_cookie != '':
            try:
                model_state = json.loads(model_state_cookie)
            except:
                response = make_response(error_report(
                    "For some reason, we could not parse the cookie for this site. " +
                    "We just tried to clear it, but if the error persists clear the cookie and try again."))
                return response_set_model_cookie(response, dict())
        else:
            model_state = dict()
        model_state['model'] = model

        try:
            # attempt to get the variable list
            variables, right_sides = get_model_variables(model)
        except Exception as e:
            # respond with an error message if submission is ill-formed
            response = make_response(error_report(str(e)))
            return response_set_model_cookie(response, model_state)

        # cleanup the model
        model = ""
        for variable, rhs in zip(variables, right_sides):
            model += "{v} = {r}\n".format(v=variable, r=rhs)
        model_state['model'] = model

        # respond with the options page
        response = make_response(render_template('options.html', variables=variables))
        return response_set_model_cookie(response, model_state)

    def get_model_variables(model):
        variables = []
        right_sides = []
        too_many_eq_msg = "Count of ='s on line {lineno} was {eq_count} but each line must have a single = sign."
        zero_len_var_msg = "No varible found before = on line {lineno}."
        zero_len_rhs_msg = "No right hand side of equation on line {lineno}."
        invalid_var_name_msg = "One line {lineno}, variable name must be alpha-numeric and include at least one letter."
        for lineno, line in enumerate(model.splitlines(), start=1):
            # check for _one_ equals sign
            if line.count('=') != 1:
                raise Exception(too_many_eq_msg.format(lineno=lineno,
                                                       eq_count=line.count('=')))
            variable, rhs = line.split('=')
            variable = variable.strip()
            rhs = rhs.strip()
            # check to see if lhs is a valid symbol. TODO: what variable names does convert.py allow?
            if len(variable) == 0:
                raise Exception(zero_len_var_msg.format(lineno=lineno))
            if not variable.isalnum() or not any(c in string.ascii_letters for c in variable):
                raise Exception(invalid_var_name_msg.format(lineno=lineno))
            variables.append(variable)
            # do _minimal_ checking on RHS
            if len(rhs) == 0:
                raise Exception(zero_len_rhs_msg.format(lineno=lineno))
            right_sides.append(rhs)
        return variables, right_sides

    def response_set_model_cookie(response, model_state):
        # set cookie expiration 90 days hence
        expire_date = datetime.datetime.now() + datetime.timedelta(days=90)
        response.set_cookie('state', json.dumps(model_state), expires=expire_date)
        return response

    def error_report(error_string):
        """ display error reports from invalid user input """
        return render_template('error.html', error_message=error_string)

    ####################################################################################################
    # startup boilerplate

    # if __name__ == '__main__':
    #    app.run()

    return app
