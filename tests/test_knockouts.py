# TODO: more
import itertools
from typing import Dict

import numpy as np

from steady_cell_phenotype._util import process_model_text


def test_knockouts():
    model_text = """
    A=B
    B=A"""

    knockouts: Dict[str, int] = {"A": 1}  # knockout A
    continuous: Dict[str, bool] = {}  # no variables continuous

    variables, update_fn, equation_system = process_model_text(
        model_text, knockouts, continuous
    )

    assert variables == ("A", "B")

    # function becomes constant on 2nd step
    for a, b in itertools.product(range(3), repeat=2):
        assert np.all(update_fn(update_fn([a, b])) == [1, 1])
