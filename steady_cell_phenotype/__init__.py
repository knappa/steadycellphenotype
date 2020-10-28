#!/usr/bin/env python3
#
# A tool to analyze ternary networks
#
# Copyright Adam C. Knapp 2019-2020
# Funded by American University Mellon Grant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tempfile

import matplotlib
from flask import Flask, request, Response, make_response
from werkzeug.utils import secure_filename

matplotlib.use('agg')

from ._util import *


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY='dev',
        MAX_CONTENT_LENGTH=512 * 1024,  # maximum upload size: 512 kilobytes
        UPLOAD_FOLDER=tempfile.TemporaryDirectory(),  # TODO: does this make sense??? doesn't work without it
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
    def download_tsv() -> Response:
        try:
            tsv = request.form['model_result'].strip()
            return Response(
                tsv + '\n',  # For parsing
                mimetype="text/tab-separated-values",
                headers={"Content-disposition": "attachment; filename=model_result.tsv"})
        except KeyError:
            response = make_response(error_report(
                "Something odd happened."))
            return response

    ####################################################################################################
    # the main / model entry page

    @app.route('/', methods=['GET', 'POST'])
    def index():
        """ render the main page """

        # get model file, if present
        model_from_file = None
        if request.method == 'POST' and 'model-file' in request.files:
            model_file = request.files['model-file']
            if model_file.filename != '':
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    filename = tmp_dir_name + '/' + secure_filename(model_file.filename)
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
    # main computational page

    @app.route('/compute/', methods=['POST'])
    def compute():
        """ render the results of computation page """

        # load existing state cookie, if it exists and makes sense
        model_state_cookie = request.cookies.get('state')
        if model_state_cookie is not None and model_state_cookie != '':
            try:
                model_state = json.loads(model_state_cookie)
            except json.JSONDecodeError:
                response = make_response(error_report(
                    "For some reason, we could not parse the cookie for this site. " +
                    "We just tried to clear it, but if the error persists clear the cookie manually and try again."))
                return response_set_model_cookie(response, dict())
        else:
            # respond with an error message if submission is ill-formed
            response = make_response(error_report(
                "We couldn't find the cookie which contains your model, " +
                "please go back to the main page and try again"))
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

        # get which variables to visualize
        visualize_variables = {variable.strip(): True if '{}-visualized'.format(variable) in request.form else False
                               for variable in variables}

        # get initial state
        init_state = dict()
        for variable in variables:
            form_name = '{}-init'.format(variable)
            if form_name in request.form and request.form[form_name] != 'None':
                init_state[variable.strip()] = request.form[form_name]

        knockout_model = ""
        for variable, rhs in zip(variables, right_sides):
            if variable in knockouts:
                knockout_model += "{v} = {r}\n".format(v=variable, r=knockouts[variable])
            else:
                knockout_model += "{v} = {r}\n".format(v=variable, r=rhs)

        # get number of iterations for the simulation
        try:
            num_iterations = int(request.form['num_samples'])
        except ValueError:
            num_iterations = 0  # not going to waste any effort on garbage

        check_nearby = 'trace-nearby-checkbox' in request.form and request.form['trace-nearby-checkbox'] == 'Yes'

        # decide which type of computation to run
        if 'action' not in request.form or request.form['action'] not in ['cycles', 'fixed_points', 'trace']:
            response = make_response(error_report(
                'The request was ill-formed, please go back to the main page and try again'))
            return response_set_model_cookie(response, model_state)
        elif request.form['action'] == 'cycles':
            return compute_cycles(model_state, knockouts, continuous, num_iterations, visualize_variables)
        elif request.form['action'] == 'fixed_points':
            return compute_fixed_points(model_state, knockout_model, variables, continuous)
        elif request.form['action'] == 'trace':
            return compute_trace(model_state, knockout_model, variables, continuous, init_state, check_nearby)
        else:
            return str(request.form)

    ####################################################################################################

    from ._compute_fixed_points import compute_fixed_points

    from ._compute_cycles import compute_cycles

    from ._compute_trace import compute_trace

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
            except json.JSONDecodeError:
                response = make_response(error_report(
                    "For some reason, we could not parse the cookie for this site. " +
                    "We just tried to clear it, but if the error persists clear the cookie manually and try again."))
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

    ####################################################################################################
    # startup boilerplate

    return app
