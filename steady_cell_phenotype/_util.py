import datetime
import html
import json
import string

from flask import render_template

MAX_SUPPORTED_VARIABLE_STATES = 6

def html_encode(msg):
    if type(msg) is not str:
        msg = msg.decode()
    return html.escape(msg).replace('\n', '<br>').replace(' ', '&nbsp')


def error_report(error_string):
    """ display error reports from invalid user input """
    return render_template('error.html', error_message=error_string)


def response_set_model_cookie(response, model_state):
    # set cookie expiration 90 days hence
    expire_date = datetime.datetime.now() + datetime.timedelta(days=90)
    response.set_cookie('state', json.dumps(model_state), expires=expire_date)
    return response


def get_model_variables(model):
    variables = []
    right_sides = []
    too_many_eq_msg = "Count of ='s on line {lineno} was {eq_count} but each line must have a single = sign."
    zero_len_var_msg = "No variable found before = on line {lineno}."
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


def decode_int(coded_value, num_variables):
    """ Decode long-form int into trinary """
    if isinstance(coded_value, str):
        if coded_value[:2] == '0x':
            coded_value = int(coded_value, 16)
        else:
            coded_value = int(coded_value)
    exploded_values = []
    for _ in range(num_variables):
        next_value = coded_value % 3
        exploded_values.append(next_value)
        coded_value = (coded_value - next_value) // 3
    exploded_values.reverse()
    return exploded_values
