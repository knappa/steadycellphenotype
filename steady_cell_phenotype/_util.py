import datetime
import html
import itertools
import json
import string
from typing import Dict, List
import numpy as np
from attr import attrs, attrib
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


@attrs
class StreamingStats(object):
    mean: float = attrib(init=False, default=float('nan'))
    scaled_var: float = attrib(init=False, default=float('nan'))
    var: float = attrib(init=False, default=float('nan'))
    count: int = attrib(init=False, default=0)

    def add(self, datum: float):
        self.count += 1
        if self.count == 1:
            self.mean = datum
            self.scaled_var = 0
            self.var = 0
        else:
            old_mean = self.mean
            self.mean += (datum - old_mean) / self.count
            self.scaled_var += (datum - old_mean) * (datum - self.mean)

        if self.count > 1:
            self.var = self.scaled_var / (self.count - 1)
        else:
            self.var = 0.0


@attrs
class BinCounter(object):
    bins: np.ndarray = attrib(init=False, default=np.zeros(10, dtype=np.int))
    max: int = attrib(init=False, default=-1)

    def add(self, datum: int):
        if datum >= len(self.bins):
            old_bins = self.bins
            self.bins = np.zeros(int(1.5 * datum), dtype=np.int)
            self.bins[:len(old_bins)] = old_bins
        self.bins[datum] += 1
        self.max = max(self.max, datum)

    def trimmed_bins(self):
        return np.trim_zeros(filt=self.bins, trim='b')


class HashableNdArray(object):
    """
    A wrapper to make numpy based state arrays hashable
    """
    array: np.ndarray

    def __init__(self, array):
        self.array = array

    def __repr__(self):
        return 'HashableNdArray(array=' + repr(self.array) + ')'

    def __str__(self):
        return str(self.array)

    def __getitem__(self, item):
        return self.array[item]

    def __eq__(self, other: 'HashableNdArray'):
        # decided on array_equiv rather than array_equal, because I don't want to worry about
        # the case where the shape is (1,n) or (n,1) and being compared to (n,) or some
        # other such nonsense.
        return np.array_equiv(self.array, other.array)

    def __hash__(self):
        # view the array as a ternary expansion; a bijective hash
        return int(np.dot(np.vander([3], len(self.array)), self.array))


def complete_search_generator(variables: List[str], constants_vals: Dict[str, int]):
    """
    Generator which yields all possible states, with constant variables fixed

    Returns
    -------
    Generator[Tuple]
    """
    constants = tuple(constants_vals.keys())
    num_non_constant_vars = len(variables) - len(constants)
    non_constant_vars = list(set(variables) - set(constants))
    for var_dict in map(lambda tup: dict(zip(tup[0], tup[1])),
                        zip(itertools.repeat(non_constant_vars),
                            itertools.product(range(3), repeat=num_non_constant_vars))):
        state = tuple(var_dict[var] if var in var_dict else constants_vals[var] for var in variables)
        yield state


def random_search_generator(num_iterations, variables: List[str], constants_vals: Dict[str, int]):
    """
    Generates `num_iterations` random states, with constant variables fixed.

    Parameters
    ----------
    num_iterations
    variables
    constants_vals

    Returns
    -------
    Generator[Tuple]
    """
    constants = tuple(constants_vals.keys())
    for _ in range(num_iterations):
        # making the choice to start constant variables at their constant values
        init_state_dict = {var: np.random.randint(3) if var not in constants else constants_vals[var]
                           for var in variables}
        state = tuple(init_state_dict[var] for var in variables)
        yield state
