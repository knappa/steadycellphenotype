from steady_cell_phenotype.equation_system import EquationSystem
from steady_cell_phenotype.poly import Monomial

TEST_SYSTEMS = [
    """
    A=B
    B=A
    """,
]


def test_roundtrip():
    for system in TEST_SYSTEMS:
        eqn_sys = EquationSystem.from_text(system.strip())
        sbml_sys = str(eqn_sys.as_sbml_qual())
        assert str(EquationSystem.from_sbml_qual(sbml_sys)).strip() == system.strip()


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
