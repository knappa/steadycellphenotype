# TODO

import itertools
from typing import Dict

import numpy as np

from steady_cell_phenotype._util import process_model_text


def test_update_fn1():
    model_text = """
    A=B
    B=A"""

    knockouts: Dict[str, int] = {}  # no knockout
    continuous: Dict[str, bool] = {}  # all variables continuous

    variables, update_fn, equation_system = process_model_text(
        model_text, knockouts, continuous
    )

    assert variables == ("A", "B")

    # function is order 2
    for a, b in itertools.product(range(3), repeat=2):
        assert np.all(update_fn(update_fn([a, b])) == [a, b])


def test_update_fn2():
    model_text = """
    A=B
    B=C
    C=A"""

    knockouts: Dict[str, int] = {}  # no knockout
    continuous: Dict[str, bool] = {}  # all variables continuous

    variables, update_fn, equation_system = process_model_text(
        model_text, knockouts, continuous
    )

    assert variables == ("A", "B", "C")

    # function is order 3
    for a, b, c in itertools.product(range(3), repeat=3):
        assert np.all(update_fn(update_fn(update_fn([a, b, c]))) == [a, b, c])
