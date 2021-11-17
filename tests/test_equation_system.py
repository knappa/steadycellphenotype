# TODO: more
import itertools
from typing import Dict

import numpy as np

from steady_cell_phenotype._util import process_model_text


def test_parse():
    model_text = """
    A=B
    B=A"""

    knockouts = {}  # no knockout
    continuous: Dict[str, bool] = {}  # no continuous variables

    variables, update_fn, equation_system = process_model_text(
        model_text, knockouts, continuous
    )

    assert variables == ("A", "B")

    for a, b in itertools.product(range(3), repeat=2):
        assert np.all(update_fn([a, b]) == [b, a])
