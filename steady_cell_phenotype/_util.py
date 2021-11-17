import html
import importlib.resources
import itertools
import pathlib
import re
import string
from typing import Callable, Dict, Iterator, List, Tuple, Union

import numba
import numpy as np
from attr import attrib, attrs
from flask import render_template

from steady_cell_phenotype.equation_system import EquationSystem

MAX_SUPPORTED_VARIABLE_STATES = 6


def get_resource_path(filename: str) -> pathlib.Path:
    """
    Obtain the location of steady_cell_phenotype's data files regardless of package location.

    Parameters
    ----------
    filename: str
        File to obtain
    Returns
    -------
    File's contents as a string
    """
    # See: https://docs.python.org/3.7/library/importlib.html#module-importlib.resources
    with importlib.resources.path("steady_cell_phenotype", filename) as path:
        return path


def get_text_resource(filename: str) -> str:
    """
    Obtain the contents of steady_cell_phenotype's data files regardless of package location.

    Parameters
    ----------
    filename: str
        File to obtain
    Returns
    -------
    File's contents as a string
    """
    # See: https://docs.python.org/3.7/library/importlib.html#module-importlib.resources
    with importlib.resources.open_text(
        "steady_cell_phenotype", filename
    ) as file_handle:
        data = file_handle.read()
    return data


def html_encode(msg):
    if type(msg) is not str:
        msg = msg.decode()
    return html.escape(msg).replace("\n", "<br>").replace(" ", "&nbsp")


def error_report(error_string):
    """display error reports from invalid user input"""
    return render_template("error.html", error_message=error_string)


def message(error_string):
    """display error reports from invalid user input"""
    return render_template("message.html", message=error_string)


def get_model_variables(model) -> Tuple[List[str], List[str]]:
    variables: List[str] = []
    right_sides: List[str] = []
    too_many_eq_msg = (
        "Count of ='s on line {lineno} was {eq_count} but each"
        " line must have a single = sign."
    )
    zero_len_var_msg = "No variable found before = on line {lineno}."
    zero_len_rhs_msg = "No right hand side of equation on line {lineno}."
    invalid_var_name_msg = (
        "One line {lineno}, variable name must be alpha-numeric (plus underscore), "
        "and begin with a letter."
    )
    for lineno, line in enumerate(model.splitlines(), start=1):
        # check for _one_ equals sign
        if line.count("=") != 1:
            raise Exception(
                too_many_eq_msg.format(lineno=lineno, eq_count=line.count("="))
            )
        variable: str
        rhs: str
        variable, rhs = line.split("=")
        variable = variable.strip()
        rhs = rhs.strip()
        # check to see if lhs is a valid symbol.
        # TODO: what variable names does scp_converter.py allow?
        if len(variable) == 0:
            raise Exception(zero_len_var_msg.format(lineno=lineno))
        if not re.match(r"^\w+$", variable) or variable[0] not in string.ascii_letters:
            raise Exception(invalid_var_name_msg.format(lineno=lineno))
        variables.append(variable)
        # do _minimal_ checking on RHS
        if len(rhs) == 0:
            raise Exception(zero_len_rhs_msg.format(lineno=lineno))
        right_sides.append(rhs)
    return variables, right_sides


def decode_int(coded_value, num_variables):
    """Decode long-form int into trinary"""
    if isinstance(coded_value, str):
        if coded_value[:2] == "0x":
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
    """
    Implements Welford's online algorithm for mean and variance calculation. See Knuth Vol 2, pg 232
    """

    mean: float = attrib(init=False, default=float("nan"))
    scaled_var: float = attrib(init=False, default=float("nan"))
    var: float = attrib(init=False, default=float("nan"))
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
    """
    Utility class for streaming bincounts.
    """

    bins: np.ndarray = attrib(init=False, default=np.zeros(0, dtype=int))
    max: int = attrib(init=False, default=-1)

    def total(self) -> int:
        return int(np.sum(self.bins))

    def add(self, datum: Union[int, "BinCounter"]):
        if isinstance(datum, int):
            if datum >= len(self.bins):
                old_bins = self.bins
                self.bins = np.zeros(1 + int(1.5 * datum), dtype=int)
                self.bins[: len(old_bins)] = old_bins
            self.bins[datum] += 1
            self.max = max(self.max, datum)
        elif isinstance(datum, BinCounter):
            new_max = max(self.max, datum.max)
            new_bins = np.zeros(shape=(new_max + 1))
            new_bins[: self.max + 1] = self.bins[: self.max + 1]
            new_bins[: datum.max + 1] += datum.bins[: datum.max + 1]
            self.max = new_max
            self.bins = new_bins
        else:
            raise RuntimeError("Cannot handle this.")

    def trimmed_bins(self):
        return np.trim_zeros(filt=self.bins, trim="b")


@numba.njit
def ternary_hash(arr: np.ndarray) -> int:
    accumulator: int = 0
    for idx in range(len(arr)):
        accumulator = 3 * accumulator + arr[idx]
    return int(accumulator)


class HashableNdArray(object):
    """
    A wrapper to make numpy based state arrays hashable
    """

    array: np.ndarray
    hash: int

    def __init__(self, array):
        self.array = array
        # view the array as a ternary expansion; a bijective hash
        self.hash = ternary_hash(array)

    def __repr__(self):
        # return 'HashableNdArray(array=' + repr(self.array) + ')'
        return repr(self.array)

    def __str__(self):
        return str(self.array)

    def __getitem__(self, item):
        return self.array[item]

    def __eq__(self, other: "HashableNdArray"):
        # decided on array_equiv rather than array_equal, because I don't want to worry about
        # the case where the shape is (1,n) or (n,1) and being compared to (n,) or some
        # other such nonsense.
        #
        # note: as this is a performance critical point, I am not going to check the type of 'other'
        # and instead trust that I don't do anything brain-dead.
        return self.hash == other.hash and np.array_equiv(self.array, other.array)

    def __hash__(self):
        return self.hash


def complete_search_generator(
    variables: List[str], constants_vals: Dict[str, int]
) -> Iterator[np.ndarray]:
    """
    Generator which yields all possible states, with constant variables fixed

    Returns
    -------
    Iterator[np.ndarray]
    """
    constants = tuple(constants_vals.keys())
    num_non_constant_vars = len(variables) - len(constants)
    non_constant_vars = list(set(variables) - set(constants))
    for var_dict in map(
        lambda tup: dict(zip(tup[0], tup[1])),
        zip(
            itertools.repeat(non_constant_vars),
            itertools.product(range(3), repeat=num_non_constant_vars),
        ),
    ):
        state = np.fromiter(
            (
                var_dict[var] if var in var_dict else constants_vals[var]
                for var in variables
            ),
            dtype=np.int64,
        )
        yield state


def random_search_generator(
    num_iterations,
    variables: List[str],
    constants_vals: Dict[str, int],
    batch_size=2000,
) -> Iterator[np.ndarray]:
    """
    Generates `num_iterations` random states, with constant variables fixed.

    Parameters
    ----------
    num_iterations : int
        number of iterations to generate
    variables : List[str]
        ordered list of variables
    constants_vals : Dict[str, int]
        dictionary of variables with their constant iteration
    batch_size : int
        size of internal batch, for performance tuning, no other external effect

    Returns
    -------
    Iterator[np.ndarray]
    """
    num_variables = len(variables)
    constant_indices = np.array(
        [idx for idx, var in enumerate(variables) if var in constants_vals],
        dtype=np.int64,
    )
    constant_val_arr = np.array(
        [constants_vals[var] for var in variables if var in constants_vals],
        dtype=np.int64,
    )
    count = 0
    while count < num_iterations:
        init_states = np.random.randint(3, size=(batch_size, num_variables), dtype=int)
        # set constant values
        init_states[:, constant_indices] = constant_val_arr

        for idx in range(batch_size):
            yield init_states[idx, :]
            count += 1
            if count >= num_iterations:
                return


def batcher(
    state_gen: Iterator[np.ndarray], variables, batch_size
) -> Iterator[np.ndarray]:
    num_variables = len(variables)
    batch: np.ndarray
    try:
        while True:
            batch = np.zeros(shape=(batch_size, num_variables), dtype=int)
            for idx in range(batch_size):
                batch[idx] = next(state_gen)
            yield batch
    except StopIteration:
        # noinspection PyUnboundLocalVariable
        yield batch[:idx, :]


def get_phased_trajectory(
    init_state: np.ndarray, update_fn: Callable
) -> Tuple[np.ndarray, HashableNdArray]:
    """
    evolve an initial state until it reaches a limit cycle

    Parameters
    ----------
    init_state
    update_fn

    Returns
    -------
    trajectory, phase-point pair
    """
    state = init_state
    trajectory = list()
    trajectory_set = set()  # set lookup should be faster

    # compute state by state until we have a repeat
    hashable_state = HashableNdArray(state)
    while hashable_state not in trajectory_set:
        trajectory.append(hashable_state)
        trajectory_set.add(hashable_state)
        state = update_fn(state)
        hashable_state = HashableNdArray(state)

    # separate trajectory into in-bound and limit-cycle parts
    repeated_state = HashableNdArray(state)
    repeated_state_index = trajectory.index(repeated_state)
    limit_cycle = trajectory[repeated_state_index:]

    # find state in limit cycle with smallest hash (i.e. smallest lexicographic
    # ordering if there is no integer overflow)
    # this is our phase fixing point
    cycle_min_index: int = 0
    cycle_min: int = hash(limit_cycle[0])
    for idx in range(1, len(limit_cycle)):
        nxt_hash: int = hash(limit_cycle[idx])
        if nxt_hash < cycle_min:
            cycle_min_index = idx
            cycle_min = nxt_hash

    # get trajectory with phase
    phase_idx: int = len(trajectory) - len(limit_cycle) + cycle_min_index
    phased_trajectory = np.array(
        [hashable.array for hashable in trajectory[:phase_idx]], dtype=np.int64
    )

    return phased_trajectory, trajectory[phase_idx]


def get_trajectory(
    init_state: np.ndarray, update_fn: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    evolve an initial state until it reaches a limit cycle

    Parameters
    ----------
    init_state
    update_fn

    Returns
    -------
    trajectory, limit cycle
    """
    state = init_state
    trajectory = list()
    trajectory_set = set()  # set lookup should be faster

    # compute state by state until we have a repeat
    hashable_state = HashableNdArray(state)
    while hashable_state not in trajectory_set:
        trajectory.append(hashable_state)
        trajectory_set.add(hashable_state)
        state = update_fn(state)
        hashable_state = HashableNdArray(state)

    # separate trajectory into in-bound and limit-cycle parts
    repeated_state = HashableNdArray(state)
    repeated_state_index = trajectory.index(repeated_state)

    trimmed_trajectory = np.array(
        [hashable.array for hashable in trajectory[:repeated_state_index]]
    )
    limit_cycle = np.array(
        [hashable.array for hashable in trajectory[repeated_state_index:]]
    )

    return trimmed_trajectory, limit_cycle


def process_model_text(
    model_text: str, knockouts: Dict[str, int], continuous: Dict[str, bool]
) -> Tuple[List[str], Callable, EquationSystem]:
    equation_system = EquationSystem.from_text(model_text)
    equation_system = equation_system.continuous_functional_system(
        continuous_vars=tuple([var for var in continuous if continuous[var]])
    )
    equation_system = equation_system.knockout_system(knockouts)

    # create an update function
    variables: List[str]
    update_fn: Callable
    variables, update_fn = equation_system.as_numpy()

    return variables, update_fn, equation_system
