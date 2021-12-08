#!/usr/bin/env python3
#
# A tool to analyze ternary networks
#
# Copyright Adam C. Knapp 2019-2021
# Funded by American University Mellon Grant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import copy
import os
import tempfile
import uuid
from typing import Dict, List

import matplotlib
from bs4 import BeautifulSoup
from flask import (Flask, Response, make_response, render_template, request,
                   session)
from werkzeug.utils import secure_filename

from steady_cell_phenotype._util import (error_report, get_model_variables,
                                         process_model_text)
from steady_cell_phenotype.equation_system import ParseError
from steady_cell_phenotype.session_data_cache import SessionDataCache


def create_app(test_config=None):

    matplotlib.use("agg")

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", default="dev"),
        MAX_CONTENT_LENGTH=512 * 1024,  # maximum upload size: 512 kilobytes
        UPLOAD_FOLDER=tempfile.TemporaryDirectory(),
        # TODO: added to assist when developing, should be removed in production
        TEMPLATES_AUTO_RELOAD=True,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    data_cache: SessionDataCache = SessionDataCache(size=1000)

    ################################################################################################
    # some _effectively_ static pages, which share a basic template

    # noinspection PyUnusedLocal
    @app.errorhandler(404)
    def page_not_found(error):
        return render_template("page_not_found.html"), 404

    @app.route("/about/")
    def about():
        """render the about page"""
        return render_template("about.html")

    @app.route("/quickstart/")
    def quickstart():
        """render the quick start page"""
        return render_template("quickstart.html")

    @app.route("/docs/")
    def docs():
        """render the documentation page"""
        return render_template("docs.html")

    @app.route("/source/")
    def source():
        """render the links-to-source page"""
        return render_template("source.html")

    ################################################################################################
    # a download page

    @app.route("/download-csv/", methods=["POST"])
    def download_csv() -> Response:
        try:
            csv = request.form["model_result"].strip()
            return Response(
                csv + "\n",  # For parsing
                mimetype="text/comma-separated-values",
                headers={
                    "Content-disposition": "attachment; filename=model_result.csv"
                },
            )
        except KeyError:
            response = make_response(error_report("Something odd happened."))
            return response

    ################################################################################################
    # a model download page

    @app.route("/download-model/<string:modeltype>", methods=["POST"])
    def download_model(modeltype: str) -> Response:
        use_sbml_qual_format = modeltype[-5:] == ".sbml"
        use_text_format = modeltype[-4:] == ".txt"

        # get the variable list and right hand sides
        model_text: str = request.form["model_text"].strip()

        variables, update_fn, submitted_equation_system = process_model_text(
            model_text, dict(), dict()
        )

        if use_text_format:
            response = make_response(str(submitted_equation_system))
            response.headers["Content-Type"] = "text/plain"
            return response

        if use_sbml_qual_format:
            sbml_qual: BeautifulSoup = submitted_equation_system.as_sbml_qual()
            response = make_response(str(sbml_qual.prettify()))
            response.headers["Content-Type"] = "application/xml"
            return response

        return make_response(error_report("Unknown Error"))

    ################################################################################################
    # the main / model entry page

    @app.route("/", methods=["GET", "POST"])
    def index():
        """render the main page"""

        # get model file, if present
        model_from_file = None
        if request.method == "POST" and "model-file" in request.files:
            model_file = request.files["model-file"]
            if model_file.filename != "":
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    filename = tmp_dir_name + "/" + secure_filename(model_file.filename)
                    model_file.save(filename)
                    with open(filename, "r") as file:
                        model_from_file = file.read().strip()

                    from steady_cell_phenotype.equation_system import \
                        EquationSystem

                    sbml_model = None
                    try:
                        sbml_model = EquationSystem.from_sbml_qual(model_from_file)
                    except ParseError:
                        pass
                    # SBML parse returns a string when it hits an error
                    if sbml_model is not None and not isinstance(sbml_model, str):
                        model_from_file = str(sbml_model)

        session_uuid: str = get_session_uuid()

        if model_from_file is not None:
            data_cache[session_uuid] = update_session_data(
                data_cache[session_uuid], model_text=model_from_file
            )

        model_text = (
            data_cache[session_uuid]["model_text"]
            if "model_text" in data_cache[session_uuid]
            else ""
        )
        model_lines = model_text.count("\n")
        return render_template(
            "index.html", model_text=model_text, rows=max(10, model_lines + 2)
        )

    def update_session_data(
        previous_session_data: Dict, *, model_text: str = None
    ) -> Dict:
        if previous_session_data is None:
            return {"model_text": model_text}
        else:
            previous_session_data = copy.deepcopy(previous_session_data)
            if model_text is not None:
                previous_session_data["model_text"] = model_text
            return previous_session_data

    def get_session_uuid() -> str:
        if session.new or "uuid" not in session:
            session.permanent = True
            session["uuid"] = str(uuid.uuid1())
        if str(session["uuid"]) not in data_cache:
            data_cache[str(session["uuid"])] = update_session_data({})
        return session["uuid"]

    ################################################################################################

    # main computational page

    @app.route("/compute/", methods=["POST"])
    def compute():
        """render the results of computation page"""

        session_uuid = get_session_uuid()
        if "model_text" not in data_cache[session_uuid]:
            return make_response(
                error_report("Please go to the main page to enter your model.")
            )

        # get the variable list and right hand sides
        model_text: str = data_cache[session_uuid]["model_text"]
        variables: List[str]
        right_sides: List[str]
        try:
            # attempt to get the variable list
            variables, right_sides = get_model_variables(model_text)
        except Exception as e:
            # respond with an error message if submission is ill-formed
            return make_response(error_report(str(e)))

        # decide which variables the user specified as continuous
        continuous: Dict[str, bool] = {
            variable.strip(): True
            if "{}-continuous".format(variable) in request.form
            else False
            for variable in variables
        }

        # create knockout model i.e. set certain values to constants without deleting the formulae
        knockouts: Dict[str, int] = {
            variable.strip(): int(request.form["{}-KO".format(variable)])
            for variable in variables
            if request.form["{}-KO".format(variable)] != "None"
        }

        # get which variables to visualize
        visualize_variables: Dict[str, bool] = {
            variable.strip(): True
            if "{}-visualized".format(variable) in request.form
            else False
            for variable in variables
        }

        # TODO: equation-system supports this now, should move to that.
        knockout_model: str = ""
        for variable, rhs in zip(variables, right_sides):
            if variable in knockouts:
                knockout_model += "{v} = {r}\n".format(
                    v=variable, r=knockouts[variable]
                )
            else:
                knockout_model += "{v} = {r}\n".format(v=variable, r=rhs)

        # decide which type of computation to run
        if "action" not in request.form or request.form["action"] not in [
            "cycles",
            "fixed_points",
            "trace",
        ]:
            return make_response(
                error_report(
                    "The request was ill-formed, please go back to the main page and try again"
                )
            )
        elif request.form["action"] == "cycles":
            # get number of iterations for the simulation
            try:
                num_iterations = int(request.form["num_samples"])
            except ValueError:
                return make_response(
                    error_report(
                        "The request was ill-formed, please go back to the main page and try again"
                    )
                )

            return compute_cycles(
                model_text=model_text,
                knockouts=knockouts,
                continuous=continuous,
                num_iterations=num_iterations,
                visualize_variables=visualize_variables,
            )
        elif request.form["action"] == "fixed_points":
            return compute_fixed_points(knockout_model, variables, continuous)
        elif request.form["action"] == "trace":
            # get initial state
            init_state: Dict[str, str] = dict()
            for variable in variables:
                form_name = "{}-init".format(variable)
                if form_name in request.form and request.form[form_name] != "None":
                    init_state[variable.strip()] = request.form[form_name]

            # check if we should inspect nearby states
            check_nearby: bool = (
                "trace-nearby-checkbox" in request.form
                and request.form["trace-nearby-checkbox"] == "Yes"
            )

            return compute_trace(
                model_text=model_text,
                knockouts=knockouts,
                continuous=continuous,
                init_state=init_state,
                visualize_variables=visualize_variables,
                check_nearby=check_nearby,
            )
        else:
            return str(request.form)

    ################################################################################################

    from steady_cell_phenotype._compute_cycles import compute_cycles
    from steady_cell_phenotype._compute_fixed_points import \
        compute_fixed_points
    from steady_cell_phenotype._compute_trace import compute_trace

    ################################################################################################
    # model options page

    @app.route("/options/", methods=["POST"])
    def options():
        """model options page"""

        session_uuid = get_session_uuid()

        # get submitted model from Form
        model_text = request.form["model"].strip()

        try:
            # attempt to get the variable list
            variables, right_sides = get_model_variables(model_text)
        except Exception as e:
            # respond with an error message if submission is ill-formed
            return make_response(error_report(str(e)))

        # cleanup the model
        model = ""
        for variable, rhs in zip(variables, right_sides):
            model += "{v} = {r}\n".format(v=variable, r=rhs)

        data_cache[session_uuid] = update_session_data(
            data_cache[session_uuid], model_text=model
        )

        # respond with the options page
        return make_response(render_template("options.html", variables=variables))

    ################################################################################################
    # startup boilerplate

    return app
