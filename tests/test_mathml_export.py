from itertools import product
from typing import Tuple

from steady_cell_phenotype.equation_system import EquationSystem
from steady_cell_phenotype.poly import Monomial

TEST_SYSTEMS = [
    """
    A=B
    B=A
    """,
    """
    A=MAX(A,B)
    B=NOT(A)
    """,
    """
    A=MIN(A,B)
    B=NOT(A)
    """,
    """
    A=MAX(A,NOT(B))
    B=MIN(NOT(A),B)
    """,
]


def test_roundtrip():
    for system in TEST_SYSTEMS:
        eqn_sys = EquationSystem.from_text(system.strip())
        sbml_sys = EquationSystem.from_sbml_qual(str(eqn_sys.as_sbml_qual()))

        system_vars: Tuple[str] = eqn_sys.target_variables()
        for vals in product(range(3), repeat=len(system_vars)):
            params = dict(zip(system_vars, vals))
            eqn_eval = eqn_sys.eval(params)
            sbml_eval = sbml_sys.eval(params)
            assert all(eqn_eval[var] == sbml_eval[var] for var in system_vars)


def test_inner_mathml():
    x = Monomial.as_var("x")
    y = Monomial.as_var("y")

    assert str(x._make_inner_mathml()) == "<ci>x</ci>"
    assert (
        str((x + 1)._make_inner_mathml())
        == '<apply><plus/><ci>x</ci><cn type="integer">1</cn></apply>'
    )
    assert (
        str((x + y)._make_inner_mathml())
        == "<apply><plus/><ci>x</ci><ci>y</ci></apply>"
    )
