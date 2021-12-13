from typing import Dict, List

from flask import make_response, render_template

from steady_cell_phenotype.equation_system import EquationSystem


def compute_fixed_points(
    knockout_model: str, variables: List[str], continuous: Dict[str, bool]
):
    """Run the fixed-point finding computation"""

    equation_system = EquationSystem.from_text(knockout_model)
    continuous_system = equation_system.continuous_functional_system(
        continuous_vars=[variable for variable in variables if continuous[variable]]
    )

    fixed_points_dict: List[Dict[str, int]] = continuous_system.find_all_fixed_points()

    fixed_points: List[List[int]] = list(
        map(lambda fp: [fp[var] for var in variables], fixed_points_dict)
    )

    # respond with the results-of-computation page
    return make_response(
        render_template(
            "compute-fixed-points.html", variables=variables, fixed_points=fixed_points
        )
    )
