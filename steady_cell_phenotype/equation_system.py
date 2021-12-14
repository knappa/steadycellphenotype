from __future__ import annotations

import operator
from copy import deepcopy
from functools import partial, reduce
from html import escape
from itertools import product
from math import floor
from typing import (Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Union, cast)

import numpy as np
from attr import attrib, attrs
from bs4 import BeautifulSoup
from bs4.element import ResultSet, Tag

from steady_cell_phenotype.poly import (Expression, ExpressionOrInt, Function,
                                        Mod3Poly, Monomial, TruthTable,
                                        inner_mathml_constant,
                                        inner_mathml_variable, is_integer,
                                        mod_3)

UNIVARIATE_FUNCTIONS = ["NOT"]
BIVARIATE_FUNCTIONS = ["MAX", "MIN", "CONT"]

DEFAULT_COMPARTMENT_NAME = "compartment1"

FORCE_SBML_POLYNOMIAL = False
SBML_TRUTHTABLE_OUTPUT = True
DEBUG_TRUTH_TABLES = False

try:
    # noinspection PyUnresolvedReferences
    import multiprocessing

    PARALLEL = True
except ImportError:
    PARALLEL = False


####################################################################################################


class ParseError(Exception):
    """Exception raised when we go down a wrong path in parsing

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


####################################################################################################


def tokenize(input_string: str) -> Sequence[Tuple[str, Union[str, int]]]:
    """This tokenizer takes a string and tokenizes it into a list of 2-tuples. Each
    tuple is of the form (TOKEN_TYPE, VALUE) where TOKEN_TYPE can be one of OPEN_PAREN,
    CLOSE_PAREN, COMMA, EQUALS, PLUS, MINUS, TIMES, EXP, FUNCTION, or SYMBOL. VALUE only contains
    non-redundant information in the case of SYMBOL, in which is contains the string
    representation of the symbol.

    This method can fail for invalid equations, throwing an exception.

    """
    tokenized_list = []
    whitespace = [" ", "\t"]
    punctuation = ["(", ")", ",", "=", "+", "-", "*", "^"]
    function_names = ["MAX", "MIN", "NOT"]
    while len(input_string) > 0:
        if input_string[0] in whitespace:
            # remove whitespace: spaces and tabs
            input_string = input_string[1:]
        elif input_string[0] == "(":
            tokenized_list.append(("OPEN_PAREN", "("))
            input_string = input_string[1:]
        elif input_string[0] == ")":
            tokenized_list.append(("CLOSE_PAREN", ")"))
            input_string = input_string[1:]
        elif input_string[0] == ",":
            tokenized_list.append(("COMMA", ","))
            input_string = input_string[1:]
        elif input_string[0] == "=":
            tokenized_list.append(("EQUALS", "="))
            input_string = input_string[1:]
        elif input_string[0] == "+":
            tokenized_list.append(("PLUS", "+"))
            input_string = input_string[1:]
        elif input_string[0] == "-":
            tokenized_list.append(("MINUS", "-"))
            input_string = input_string[1:]
        elif input_string[0] == "*":
            tokenized_list.append(("TIMES", "*"))
            input_string = input_string[1:]
        elif input_string[0] == "^":
            tokenized_list.append(("EXP", "^"))
            input_string = input_string[1:]
        else:
            # could be a symbol, function name, or constant. can be of variable length, terminated
            # by punctuation or whitespace
            index = 0
            while (
                index < len(input_string)
                and not input_string[index] in punctuation
                and not input_string[index] in whitespace
            ):
                index += 1

            if index > 0:
                if index == 3 and input_string[0:3].upper() in function_names:
                    tokenized_list.append(("FUNCTION", input_string[0:3].upper()))
                    input_string = input_string[3:]
                else:
                    try:
                        # check to see if this is a constant.
                        const = int(input_string[0:index])
                        tokenized_list.append(("CONSTANT", const))
                    except ValueError:
                        # if it isn't parsable as an int, it is a symbol
                        tokenized_list.append(("SYMBOL", input_string[0:index]))
                    input_string = input_string[index:]
            else:
                raise Exception("Error in tokenization, cannot understand what this is")

    return tokenized_list


####################################################################################################


def find_all(find_token_string, formula):
    """generator which yields all indices (in order) of occurrences of a particular type
    (find_token_string) of token in formula, which is an enumerable of (token,value) pairs

    will also work if find_token_string is a list of strings
    """
    if type(find_token_string) == str:
        for n, (token_string, _) in enumerate(formula):
            if token_string == find_token_string:
                yield n
    else:
        # will work with lists, but we don't check it
        for n, (token_string, _) in enumerate(formula):
            if token_string in find_token_string:
                yield n


####################################################################################################
# recursive descent parser
####################################################################################################


def translate_to_expression(
    formula: Sequence[Tuple[str, Union[str, int]]]
) -> ExpressionOrInt:
    """
    Parse formula using recursive descent.

    Recurses left-to-right, building the formula on the way back up.

    Parameters
    ----------
    formula
        tokenized list

    Returns
    -------
    ExpressionOrInt
    """
    if len(formula) <= 0:
        raise ParseError("can't parse an empty formula")

    next_token_type, next_token_content = formula[0]

    # try to parse addition and subtraction
    for additive_index in find_all(["PLUS", "MINUS"], formula):
        try:
            argument_one = translate_to_expression(formula[0:additive_index])
            argument_two = translate_to_expression(formula[additive_index + 1 :])

            # we succeed!
            if formula[additive_index][0] == "PLUS":
                return argument_one + argument_two
            else:
                assert formula[additive_index][0] == "MINUS"
                return (
                    argument_one - argument_two
                )  # Note: This is, in some sense, wrong, but fixed elsewhere. TODO: Remove this
        except ParseError:
            pass

    # try to parse as a unary minus: -ARG
    if next_token_type == "MINUS":
        try:
            argument = translate_to_expression(formula[1:])
            # we succeed!
            return -argument
        except ParseError:
            pass

    # try to parse as a unary plus: +ARG
    if next_token_type == "PLUS":
        try:
            argument = translate_to_expression(formula[1:])
            # we succeed!
            return argument
        except ParseError:
            pass

    # try to parse multiplication
    for mult_index in find_all("TIMES", formula):
        try:
            argument_one = translate_to_expression(formula[0:mult_index])
            argument_two = translate_to_expression(formula[mult_index + 1 :])
            # we succeed!
            return argument_one * argument_two
        except ParseError:
            pass

    # try to parse exponentiation --- we only allow atomic integer constants in the exponent because
    # these are not to be taken mod 3
    if len(formula) >= 3 and formula[-2][0] == "EXP" and formula[-1][0] == "CONSTANT":
        exp_index = len(formula) - 2
        try:
            base = translate_to_expression(formula[0:exp_index])
            # again, we only accept constants in the exponent!
            if formula[-1][0] != "CONSTANT":
                raise ParseError("must have constant in the exponent")
            exponent = formula[-1][1]
            # we succeed!
            return base ** exponent
        except ParseError:
            pass

    # try to parse parentheses: (ARG)
    if next_token_type == "OPEN_PAREN" and formula[-1][0] == "CLOSE_PAREN":
        try:
            argument = translate_to_expression(formula[1:-1])
            # we succeed!
            return argument
        except ParseError:
            pass

    # try to parse CONST/SYMB
    if len(formula) == 1:
        if next_token_type == "CONSTANT":
            constant: int = next_token_content
            return constant
        elif next_token_type == "SYMBOL":
            symbol_text = next_token_content
            return Monomial({symbol_text: 1})
        else:
            raise ParseError("Invalid singleton")

    # try to parse functions
    if next_token_type == "FUNCTION":
        function_name = next_token_content
        # formatting sanity check; must have FUNCTION OPEN_PAREN ... CLOSE_PAREN
        if (
            len(formula) < 2
            or formula[1][0] != "OPEN_PAREN"
            or formula[-1][0] != "CLOSE_PAREN"
        ):
            raise ParseError(
                "Parse error, must have form FUNCTION OPEN_PAREN ... CLOSE_PAREN"
            )

        # lop off the function name and parentheses
        inner_formula = formula[2:-1]

        if function_name in UNIVARIATE_FUNCTIONS:
            argument = translate_to_expression(
                inner_formula
            )  # may throw ParseError, which we do not catch here
            # we succeed!
            return Function(function_name, [argument])
        elif function_name in BIVARIATE_FUNCTIONS:
            # try to find the comma
            for comma_index in find_all("COMMA", inner_formula):
                try:
                    argument_one = translate_to_expression(inner_formula[0:comma_index])
                    argument_two = translate_to_expression(
                        inner_formula[comma_index + 1 :]
                    )
                    # if we succeed, return the function
                    return Function(function_name, [argument_one, argument_two])
                except ParseError:
                    # try another comma
                    pass
            # no comma gave a valid parse
            raise ParseError("could not parse function call")
        else:
            # function name was unrecognized
            raise ParseError("This is not a function I recognize!")

    # in the case that nothing worked
    raise ParseError("Parse error!")


################################################################################################
# helper functions for parallelization, b/c Python's pickling is weirdly inflexible


def evaluate_parallel_helper(pair, mapping_dict):
    """
    Because python won't let you use lambdas in multiprocessing
    :param pair:
    :param mapping_dict:
    :return:
    """
    variable_name, expression = pair
    if isinstance(expression, int) or expression.is_constant():
        return variable_name, expression
    else:
        return variable_name, expression.full_eval(mapping_dict)


def continuity_helper(
    control_variable: str, equation: ExpressionOrInt, continuous_vars: Sequence[str]
) -> Tuple[str, ExpressionOrInt]:
    """
    Because python won't let you use lambdas in multiprocessing
    :param control_variable:
    :param equation:
    :param continuous_vars:
    :return:
    """
    if control_variable not in continuous_vars:
        return control_variable, equation
    elif isinstance(equation, int) or equation.is_constant():
        return control_variable, int(equation)
    else:
        return control_variable, equation.continuous_polynomial_version(
            control_variable
        )


def polynomial_output_parallel_helper(equation: ExpressionOrInt) -> ExpressionOrInt:
    """
    Converter from an ExpressionOrInt, to one which is a polynomial.

    Required because python won't let you use lambdas in multiprocessing

    Parameters
    ----------
    equation
        The equation to convert

    Returns
    -------
    ExpressionOrInt
    """
    if isinstance(equation, int) or equation.is_constant():
        return int(equation)
    else:
        return equation.as_polynomial()


####################################################################################################


# noinspection PyUnusedLocal
def _parse_mathml_constant_to_function(
    input_variable_list: List[str], mathml_subtag: Tag
) -> Callable:
    # constant
    if mathml_subtag.attrs["type"] != "integer":
        raise ParseError("Non-integer data type found")
    try:
        value = int(mathml_subtag.text)

        # noinspection PyUnusedLocal
        def f(input_values):
            return value

        return f
    except ValueError:
        raise ParseError(f"Could not parse {mathml_subtag.text} as an integer")


def _parse_mathml_variable_to_function(
    input_variable_list: List[str], mathml_subtag: Tag
) -> Callable:
    # variable
    variable_name = mathml_subtag.text.strip()
    try:
        variable_index = input_variable_list.index(variable_name)
    except ValueError:
        raise ParseError(
            f"'{variable_name}' not found in list"
            f" of variables: {input_variable_list}"
        )

    # noinspection PyUnusedLocal
    def f(input_values):
        return input_values[variable_index]

    return f


def _parse_mathml_function_to_function(
    input_variable_list: List[str], mathml_subtag: Tag
) -> Callable:
    tag_contents: List[Tag] = cast(
        List[Tag], list(filter(lambda x: type(x) == Tag, mathml_subtag.contents))
    )
    if len(tag_contents) <= 0:
        raise ParseError("Empty apply")
    operation = tag_contents[0].name
    operand_functions: List[Callable] = [
        _parse_mathml_to_function_helper(input_variable_list, operand_tag)
        for operand_tag in tag_contents[1:]
    ]

    # handle 'not', 'unary minus', and the modulus
    unary_operations = {
        "not": operator.not_,
        "minus": operator.neg,
        "floor": lambda x: floor(x),
    }
    if operation in unary_operations and len(operand_functions) == 1:
        operator_function = unary_operations[operation]
        operand = operand_functions[0]

        def f(input_values):
            return operator_function(operand(input_values))

        return f

    # the commutative operators
    commutative_operations = {
        "max": lambda a, b: max(a, b),
        "min": lambda a, b: min(a, b),
        "plus": operator.add,
        "add": operator.add,
        "times": operator.mul,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
        "eq": operator.eq,
        "neq": operator.ne,
    }
    if operation in commutative_operations and len(operand_functions) >= 1:
        operator_function = commutative_operations[operation]

        def f(input_values):
            return reduce(
                operator_function,
                [function(input_values) for function in operand_functions],
            )

        return f

    # remaining binary operators: comparisons and other non-commutative operations
    binary_operations = {
        "eq": operator.eq,
        "neq": operator.ne,
        "gt": operator.gt,
        "lt": operator.lt,
        "geq": operator.ge,
        "leq": operator.le,
        "minus": operator.sub,
        "divide": operator.truediv,
        "power": operator.pow,
        "rem": operator.mod,
    }
    if operation in binary_operations and len(operand_functions) == 2:
        operator_function = binary_operations[operation]
        operand_a, operand_b = operand_functions

        def f(input_values):
            return operator_function(operand_a(input_values), operand_b(input_values))

        return f

    # we didn't handle the parse
    raise ParseError(
        f"Unsupported function: {operation} on {len(operand_functions)} operands"
    )


def _parse_mathml_piecewise_to_function(
    input_variable_list: List[str], mathml_subtag: Tag
) -> Callable:
    tag_contents: List[Tag] = cast(
        List[Tag], list(filter(lambda x: type(x) == Tag, mathml_subtag.contents))
    )
    if len(tag_contents) <= 0:
        raise ParseError("Empty piecewise")

    otherwise_tags = [tag for tag in tag_contents if tag.name == "otherwise"]
    if len(otherwise_tags) > 1:
        raise ParseError("Multiple otherwise tags")
    otherwise_tag = None if len(otherwise_tags) == 0 else otherwise_tags[0]

    otherwise_function: Optional[Callable] = None
    if otherwise_tag is not None:
        otherwise_tag_contents: List[Tag] = cast(
            List[Tag], list(filter(lambda x: type(x) == Tag, otherwise_tag.contents))
        )
        if len(otherwise_tag_contents) != 1:
            raise ParseError("otherwise tag in piecewise has multiple subtags")
        otherwise_function = _parse_mathml_to_function_helper(
            input_variable_list, otherwise_tag_contents[0]
        )

    piecewise_conditions: List[Callable] = []
    piecewise_functions: List[Callable] = []
    for piece_tag in (tag for tag in tag_contents if tag.name == "piece"):
        piece_tag_contents = cast(
            List[Tag], list(filter(lambda x: type(x) == Tag, piece_tag.contents))
        )
        if len(piece_tag_contents) != 2:
            raise ParseError("<piece> tags should have exactly two subtags")
        function_tag = piece_tag_contents[0]
        condition_tag = piece_tag_contents[1]
        piecewise_functions.append(
            _parse_mathml_to_function_helper(input_variable_list, function_tag)
        )
        piecewise_conditions.append(
            _parse_mathml_to_function_helper(input_variable_list, condition_tag)
        )

    def f(input_values):
        for condition, function in zip(piecewise_conditions, piecewise_functions):
            if condition(input_values):
                return function(input_values)
        if otherwise_function:
            otherwise_function(input_values)
        else:
            raise ValueError("outside of function domain")

    return f


def _parse_mathml_to_function_helper(
    input_variable_list: List[str], mathml_subtag
) -> Callable:
    """
    Recursive building of the function from the inner mathml.

    Parameters
    ----------
    input_variable_list
        ordered list of variable names
    mathml_subtag
        inner mathml to parse to a function

    Returns
    -------
    Callable
    """
    tag_type = mathml_subtag.name
    tags: Dict[str, Callable[..., Callable]] = {
        "cn": _parse_mathml_constant_to_function,
        "ci": _parse_mathml_variable_to_function,
        "apply": _parse_mathml_function_to_function,
        "piecewise": _parse_mathml_piecewise_to_function,
    }
    if tag_type in tags:
        return tags[tag_type](input_variable_list, mathml_subtag)
    else:
        raise ParseError(f"Unsupported tag (in this position?): {tag_type}")


def parse_mathml_to_function(
    input_variables: List[str], mathml_tag: Tag
) -> Callable[..., bool]:
    # examine the contents of the mathml, there may be strings (probably \n) which we should ignore
    contents: List[Tag] = cast(
        List[Tag], list(filter(lambda x: isinstance(x, Tag), mathml_tag.contents))
    )
    if len(contents) != 1:
        raise ParseError

    return _parse_mathml_to_function_helper(input_variables, contents[0])


def parse_sbml_qual_function(
    input_variables: List[str],
    function_terms: Tag,
    max_levels: Dict[str, int],
    max_level_output: int = 2,
) -> Callable[..., int]:
    """
    Builds a function from inner mathml.

    Parameters
    ----------
    input_variables
        ordered list of variable names
    function_terms
        inner mathml to parse to a function
    max_levels: Dict[str, int]
        maximum level dictionary for free variables
    max_level_output: int
        maximum level for the output variable

    Returns
    -------
    Callable
    """
    # figure out the default level
    default_levels: ResultSet = function_terms.findChildren("qual:defaultterm")
    if len(default_levels) != 1:
        raise ParseError("There should be exactly one default level!")
    default_level_tag: Tag = default_levels[0]
    if "qual:resultlevel" not in default_level_tag.attrs:
        raise ParseError("Malformed tags")
    try:
        default_level: int = int(default_level_tag.attrs["qual:resultlevel"])
    except ValueError:
        raise ParseError(
            f"Non-integer default level {default_level_tag.attrs['qual:resultlevel']}"
        )

    # now the conditional level(s)
    conditional_levels: List[Tuple[int, Callable[..., bool]]] = []
    for conditional_level_tag in function_terms.findChildren("qual:functionterm"):
        if "qual:resultlevel" not in conditional_level_tag.attrs:
            raise ParseError("Missing qual:resultlevel")
        try:
            level: int = int(conditional_level_tag.attrs["qual:resultlevel"])
        except ValueError:
            raise ParseError("Non-int qual:resultlevel")

        level_function: Callable[..., bool] = parse_mathml_to_function(
            input_variables, conditional_level_tag.findChild("math")
        )

        conditional_levels.append((level, level_function))

    def function_realization(input_values) -> int:
        # ensure that input values don't exceed their max values
        input_values = [
            min(value, max_levels[variable])
            for value, variable in zip(input_values, input_variables)
        ]
        # find the first of the output values that registers a "true"
        for value, function in conditional_levels:
            if function(input_values):
                return value
        return default_level

    return lambda x: min(max_level_output, function_realization(x))


####################################################################################################


@attrs(init=False, slots=True, repr=False, str=False)
class EquationSystem(object):
    _formula_symbol_table: List[str] = attrib()
    _equation_dict: Dict[str, ExpressionOrInt] = attrib()
    _lines: List[Tuple[Optional[str], Optional[str]]] = attrib()
    _is_dependant: bool = attrib(default=False)

    # _lines: Tuples of variable name and comment, both optional, for printing

    def __init__(
        self,
        is_dependant: bool,
        *,
        formula_symbol_table: List[str] = None,
        equation_dict: Dict[str, ExpressionOrInt] = None,
        lines: List[Tuple[Optional[str], Optional[str]]] = None,
    ):
        if formula_symbol_table is not None and equation_dict is not None:
            self._formula_symbol_table = deepcopy(formula_symbol_table)
            self._equation_dict = deepcopy(equation_dict)
            if lines is None:
                self._lines = [
                    (symbol, None) for symbol, eqn in self._equation_dict.items()
                ]
            else:
                self._lines = [
                    (None, line)
                    if symbol is None or symbol not in self._equation_dict
                    else (symbol, line)
                    for symbol, line in lines
                ]
        elif formula_symbol_table is None and equation_dict is None:
            self._formula_symbol_table = []
            self._equation_dict = dict()
            self._lines = list()
        else:
            raise RuntimeError(
                "Must specify either an empty system,"
                " or both formula_symbol_table and equation_dict"
            )

        self._is_dependant = is_dependant

    @staticmethod
    def from_text(lines: str) -> EquationSystem:
        """
        Create a system of equations from text

        Returns
        -------
        EquationSystem
        """
        equation_system: EquationSystem = EquationSystem(is_dependant=False)
        for line_number, line in enumerate(lines.strip().splitlines()):
            equation_system.parse_and_add_equation(line, line_number=line_number + 1)
        return equation_system

    @staticmethod
    def from_sbml_qual(xml_string: str) -> Union[EquationSystem, str]:
        """
        Create a system of equations from an SBML-qual string.

        Parameters
        ----------
        xml_string: str
            SBML-qual string representation of equation system

        Returns
        -------
        EquationSystem
            if we can, otherwise an error string
        """
        soup = BeautifulSoup(xml_string, "lxml")

        species_list: List[Tag] = soup.findChildren("qual:qualitativespecies")
        # do some basic checks
        if len(species_list) == 0:
            return "No species defined in the model"
        for species in species_list:
            if "qual:id" not in species.attrs:
                return "There was a species without a name!"
            species_name = species.attrs["qual:id"]
            if len(species_name) == 0 or not species_name[0].isalpha():
                return f"Species name {species_name} is invalid!"

        symbols: List[str] = [species.attrs["qual:id"] for species in species_list]

        # record their 'maxlevel', i.e. if they are boolean or ternary, or whatever.
        # We are only going to support ternary, so error out if something else is encountered
        max_levels: Dict[str, int] = {}
        for species in species_list:
            if "qual:maxlevel" not in species.attrs:
                return (
                    f"No max level provided for {species.attrs['qual:id']}, expecting 2"
                )

            try:
                if int(species.attrs["qual:maxlevel"]) not in {1, 2}:
                    return (
                        f"Max level provided for {species.attrs['qual:id']}"
                        f" is {species.attrs['qual:maxlevel']}, expecting 1 or 2"
                    )
                max_levels[species.attrs["qual:id"]] = int(
                    species.attrs["qual:maxlevel"]
                )
            except ValueError:
                return (
                    f"Max level provided for {species.attrs['qual:id']} is"
                    f" {species.attrs['qual:maxlevel']}, cannot parse as an integer"
                )

        # some of these might be constants
        equations: Dict[str, ExpressionOrInt] = {}
        for species in species_list:
            if (
                "qual:constant" in species.attrs
                and species.attrs["qual:constant"].lower() != "false"
            ):
                target_variable = species.attrs["qual:id"].strip()
                if "qual:initiallevel" in species.attrs:
                    # if the initial level is set then we make the variable constant in the
                    # sense that the update function sets that particular value.
                    try:
                        constant_level = int(species.attrs["qual:initiallevel"])
                        equations[target_variable] = constant_level
                    except ValueError:
                        return (
                            f"Level provided for {target_variable} is"
                            f" {species.attrs['qual:initiallevel']}, cannot parse as an integer"
                        )

                else:
                    # if the initial level isn't set then we make the variable constant in the
                    # sense that the update function is the identity. Except for when the variable
                    # is boolean, where it gets a min with 1 applied.
                    if max_levels[target_variable] == 2:
                        equations[target_variable] = Monomial.as_var(target_variable)
                    else:
                        # Note: this assumes PRIME=3
                        equations[target_variable] = (
                            Monomial.as_var(target_variable) ** 2
                        )

        # now we parse the transition functions into Expressions
        for transition in soup.findChildren("qual:transition"):
            # get the target
            target_variables = [
                output["qual:qualitativespecies"]
                for output in transition.findChildren("qual:output")
            ]
            if len(target_variables) != 1:
                return "Transition functions must have exactly one output."
            target_variable = target_variables[0]

            if target_variable in equations:
                return (
                    f"Variable {target_variable} appears twice as"
                    f" the target of an update function!"
                )

            input_variables = [
                input_variable["qual:qualitativespecies"]
                for input_variable in transition.findChildren("qual:input")
            ]

            function_terms: ResultSet = transition.findChildren(
                "qual:listoffunctionterms"
            )
            if len(function_terms) != 1:
                return "Should have exactly one list of function terms"
            try:
                transition_function: Callable = parse_sbml_qual_function(
                    input_variables,
                    function_terms[0],
                    max_levels=max_levels,
                    max_level_output=max_levels[target_variable],
                )
            except ParseError as e:
                return "Could not parse functions: " + str(e)

            expression: ExpressionOrInt = 0

            def interpolation_monomial(params):
                input_value, input_variable = params
                return 1 - (Monomial.as_var(input_variable) - input_value) ** 2

            for input_values in product([0, 1, 2], repeat=len(input_variables)):
                output_value: int = transition_function(input_values)
                if output_value != 0:
                    # g(x) = sum_{c\in \F_3^n} h(c) prod_{j=0}^n (1-(x_j-c_j)**2)
                    expression += output_value * reduce(
                        operator.mul,
                        map(interpolation_monomial, zip(input_values, input_variables)),
                    )
            equations[target_variable] = expression

        for variable in symbols:
            if variable not in equations:
                return f"No update function for {variable} specified!"

        lines = [(symbol, None) for symbol in symbols]

        return EquationSystem(
            formula_symbol_table=symbols,
            equation_dict=equations,
            lines=lines,
            is_dependant=False,
        )

    def as_poly_system(self) -> EquationSystem:
        formula_symbol_table = deepcopy(self._formula_symbol_table)
        equation_dict = {
            target_var: eqn.as_polynomial()
            if isinstance(eqn, Expression)
            and not (isinstance(eqn, Mod3Poly) or isinstance(eqn, Monomial))
            else eqn
            for target_var, eqn in self._equation_dict.items()
        }

        return EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=formula_symbol_table,
            equation_dict=equation_dict,
            lines=deepcopy(self._lines),
        )

    def symbol_table(self):
        return set(self._formula_symbol_table).union(self._equation_dict.keys())

    def formula_symbol_table(self):
        return deepcopy(self._formula_symbol_table)

    def target_variables(self) -> Tuple[str]:
        return tuple(self._equation_dict.keys())

    def variables_that_vary(self) -> List[str]:
        return [
            var
            for var, eqn in self._equation_dict.items()
            if type(eqn) != int and not eqn.is_constant()
        ]

    def constant_variables(self) -> List[str]:
        return [
            var
            for var, eqn in self._equation_dict.items()
            if type(eqn) == int or eqn.is_constant()
        ]

    def consistent(self) -> bool:
        """
        Check if this makes sense as an update function.

        Returns
        -------
        True if the set of formula symbols is contained in the set of target variables
        """
        return set(self._formula_symbol_table) <= set(self._equation_dict.keys())

    ################################################################################################

    def full_eval(self, state: Dict[str, int]) -> Dict[str, int]:
        if set(state.keys()) != set(self._equation_dict.keys()):
            raise RuntimeError("Evaluating state on incorrect set of variables")
        return {
            variable: formula.eval(state)
            if isinstance(formula, Expression)
            else (formula % 3)
            if isinstance(formula, int)
            else formula
            for variable, formula in self._equation_dict.items()
        }

    def eval(self, state: Dict[str, ExpressionOrInt]) -> EquationSystem:
        return EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=deepcopy(self._formula_symbol_table),
            equation_dict={
                variable: formula.eval(state)
                if isinstance(formula, Expression)
                else mod_3(formula)
                if is_integer(formula)
                else formula
                for variable, formula in self._equation_dict.items()
            },
        )

    ################################################################################################

    def __len__(self):
        return len(self._equation_dict)

    def is_variable_dependant(self, variable: str, dependancy_order: int = 1) -> bool:
        """
        Determine if `variable` is of the form variable=const or variable=f(other_vars).

        Parameters
        ----------
        variable: str
            The variable to query
        dependancy_order: int
            maximum number of other variables we have as dependencies

        Returns
        -------
        bool
        """
        if variable not in self._equation_dict.keys():
            raise RuntimeError(f"Variable {variable} not present in this system")

        equation = self._equation_dict[variable]
        if is_integer(equation):
            return True

        eqn_variable_set = equation.get_variable_set()
        return (
            variable not in eqn_variable_set
            and len(eqn_variable_set) <= dependancy_order
        )

    def add_equation(self, variable: str, function: ExpressionOrInt):
        """
        Add an equation to the system.

        Very little sanity checking done!

        Parameters
        ----------
        variable: str
            target variable
        function: ExpressionOrInt
            the update for this

        Returns
        -------
        None
        """
        if variable in self._equation_dict:
            raise RuntimeError(
                f"{variable} already has a function in this system: \n"
                f"{self}\n{variable}\n{function}"
            )

        if is_integer(function) or function.is_constant():
            if variable not in self._formula_symbol_table:
                self._formula_symbol_table.append(variable)
        else:
            new_var_set = [variable] + list(function.get_variable_set())
            self._formula_symbol_table += [
                var for var in new_var_set if var not in self._formula_symbol_table
            ]

        self._lines.append((variable, None))
        self._equation_dict[variable] = function

    def reduce_dependant_variable(
        self,
        variable: str,
        dependancy_order: int = 1,
    ) -> Tuple[EquationSystem, ExpressionOrInt]:
        """
        Construct an EquationSystem with `variable` eliminated by substitution.

        Parameters
        ----------
        variable: str
            A variable of the system whose update is of the form variable=const or
            variable=f(other_variables).
        dependancy_order
            Specify how many variables occur in `other_variables`. Default: 1

        Returns
        -------
        reduced equation system and functional dependancies for `variable`
        """
        if not self.is_variable_dependant(variable, dependancy_order=dependancy_order):
            raise RuntimeError(
                f"Variable {variable} depends on itself or "
                f"more than {dependancy_order} other variables"
            )

        # using eval, reduced system will include the target variable which we should eliminate
        reduced_system = self.eval({variable: self._equation_dict[variable]})
        reduced_system = EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=[
                var for var in reduced_system._formula_symbol_table if var != variable
            ],
            equation_dict={
                var: eqn
                for var, eqn in reduced_system._equation_dict.items()
                if var != variable
            },
        )

        return reduced_system, self._equation_dict[variable]

    def reduce_all_dependant_variables(
        self,
        dependant_system: EquationSystem = None,
        dependancy_order: int = 1,
    ) -> Tuple[EquationSystem, EquationSystem]:
        """
        Construct an EquationSystem where all substitutable variables have been eliminated.

        Returns
        -------
        Tuple of the reduced system and the depenent system
        """
        eqn_sys = self
        dependant_system = (
            EquationSystem(is_dependant=True)
            if dependant_system is None
            else dependant_system
        )
        if not dependant_system._is_dependant:
            raise RuntimeError(
                "The dependant system isn't flagged as a dependant system"
            )

        dependant_var = ""
        while dependant_var is not None:
            dependant_var = None

            # find a dependant variable. i.e. output_var=const or output_var=f(other_vars)
            for output_var, eqn in eqn_sys:
                if (
                    isinstance(eqn, int)
                    or eqn.is_constant()
                    or eqn_sys.is_variable_dependant(
                        output_var, dependancy_order=dependancy_order
                    )
                ):
                    dependant_var = output_var
                    break

            if dependant_var is not None:
                # remove the dependant variable from the system and get its dependancy relation
                # i.e. the f(other_var) in dependant_var = f(other_vars)
                (eqn_sys, functional_dependancy,) = eqn_sys.reduce_dependant_variable(
                    dependant_var, dependancy_order=dependancy_order
                )

                # we eval the dependant system to be sure that it's still only dependant on the
                # "active" parts of the model
                dependant_system = dependant_system.eval(
                    {dependant_var: functional_dependancy}
                )
                dependant_system.add_equation(dependant_var, functional_dependancy)

        return eqn_sys, dependant_system

    def constant_system_to_dict(self) -> Dict[str, int]:
        const_dict = dict()
        for var, eqn in self._equation_dict.items():
            if is_integer(eqn):
                const_dict[var] = int(eqn)
            elif isinstance(eqn, Expression):
                if eqn.is_constant():
                    const_dict[var] = int(eqn)
                else:
                    raise RuntimeError(
                        f"Non constant expression {eqn}, cannot convert to constant dict"
                    )
            else:
                raise RuntimeError(f"Unknown value {eqn} in equation system")
        return const_dict

    def output_vars(self) -> Set[str]:
        count_as_input = {var: 0 for var in self._equation_dict.keys()}
        for var in self.symbol_table():
            for out_var, eqn in self._equation_dict.items():
                if is_integer(eqn):
                    continue
                if var in eqn.get_variable_set() and var in self._equation_dict:
                    count_as_input[var] += 1
        return {var for var, count in count_as_input.items() if count == 0}

    def separate_output_variables(self) -> Tuple[EquationSystem, EquationSystem]:
        # find all the variables which only occur as an output
        pure_outputs = self.output_vars()

        dependant_equations = {
            var: eqn for var, eqn in self._equation_dict.items() if var in pure_outputs
        }
        dependant_system = EquationSystem(is_dependant=True)
        for var, eqn in dependant_equations.items():
            dependant_system.add_equation(var, eqn)

        independant_equations = {
            var: eqn
            for var, eqn in self._equation_dict.items()
            if var not in pure_outputs
        }
        rest_of_system = EquationSystem(is_dependant=self._is_dependant)
        for var, eqn in independant_equations.items():
            rest_of_system.add_equation(var, eqn)

        return rest_of_system, dependant_system

    def find_all_fixed_points(
        self, dependant_system: EquationSystem = None
    ) -> List[Dict[str, int]]:
        if dependant_system is None:
            dependant_system = EquationSystem(is_dependant=True)
            # TODO: is the below true?
            # it may be useful to put the equatation system in polynomial form, which is canonical
            # in the sense that any formula which does not depend on a variable does not have that
            # variable as a term. We lose a little bit of pretty-printing this way, but them's the
            # breaks.
            equation_system = self.as_poly_system()
        else:
            equation_system = self

        while len(equation_system.output_vars()) > 0:
            # reduce all output variables. i.e. variables which are never 'read', only set
            functional_deps: EquationSystem
            (
                equation_system,
                functional_deps,
            ) = equation_system.separate_output_variables()

            # add all the functional deps from output variables to the dependant system
            for removed_var, eqn in functional_deps._equation_dict.items():
                dependant_system = dependant_system.eval({removed_var: eqn})
                dependant_system.add_equation(removed_var, eqn)

        # reduce all variables that can be removed by substitution
        (
            equation_system,
            functional_deps,
        ) = equation_system.reduce_all_dependant_variables(dependancy_order=1)
        # Note: we can't let this dependancy_order get too big. Otherwise the polynomials get
        # overly complicated. 1 doesn't change the "size" of the polynomial.

        if len(equation_system) == 0:
            # if variable reduction cleared everything (maybe can happen with cancellations)
            # then we use that reduced system to evaluate the dependant system. Combined,
            # they give a single fixed point.
            constants = functional_deps.constant_system_to_dict()
            evaled_system = dependant_system.eval(constants)
            constants.update(evaled_system.constant_system_to_dict())
            return [constants]

        # add all the functional deps to the dependant system
        for removed_var, eqn in functional_deps._equation_dict.items():
            dependant_system = dependant_system.eval({removed_var: eqn})
            dependant_system.add_equation(removed_var, eqn)

        # reduce everything that looks like var=f(var)
        singlets = (
            var
            for var, eqn in equation_system._equation_dict.items()
            if eqn.num_variables() == 1 and eqn.get_variable_set() == {var}
        )
        singlet = next(singlets, None)
        # if we find one of these var=f(var)
        if singlet is not None:
            found_fixed_points = []
            for value in range(3):
                evaled_system: EquationSystem = equation_system.eval({singlet: value})
                if evaled_system[singlet] == value:
                    # this will be a consistent system, at least so far
                    dependant_system_copy = dependant_system.eval({singlet: value})
                    dependant_system_copy.add_equation(singlet, value)
                    evaled_system_minus_var = EquationSystem(
                        is_dependant=self._is_dependant,
                        formula_symbol_table=[
                            var
                            for var in evaled_system._formula_symbol_table
                            if var != singlet
                        ],
                        equation_dict={
                            var: eqn
                            for var, eqn in evaled_system._equation_dict.items()
                            if var != singlet
                        },
                    )
                    found_fixed_points.extend(
                        evaled_system_minus_var.find_all_fixed_points(
                            dependant_system_copy
                        )
                    )
            return found_fixed_points

        # if you make it here, there were no singlets. So all functions are of the form
        # var = f(var_1, var_2, ... , var_n) where one of the var_i could be var.

        # We will make a trial assignment of {0,1,2} -> var, filtering the results after they are
        # returned for consistancy with update function

        # Note: this code chooses the variable to reduce based on the heuristic that clearing out
        # a highly connected variable will lead to a larger number of "easy" simplification on
        # the next round. We could probably make an even smarter choice using graph theory. i.e.
        # find a vertex which cuts
        # TODO: can networkx do anything for us here?
        sized_eqns = [
            (len(eqn.get_variable_set() | {var}), var)
            if isinstance(eqn, Expression)
            else (1, var)
            for var, eqn in equation_system._equation_dict.items()
        ]
        sized_eqns.sort(reverse=True)
        count, removed_var = sized_eqns[0]

        removed_eqn = equation_system._equation_dict[removed_var]
        partial_eqn_sys = EquationSystem(
            is_dependant=equation_system._is_dependant,
            formula_symbol_table=equation_system._formula_symbol_table,
            equation_dict={
                var: eqn
                for var, eqn in equation_system._equation_dict.items()
                if var != removed_var
            },
            lines=None,
        )

        found_fixed_points = []
        for value in range(3):
            resolved_partial_eqn_sys: EquationSystem = partial_eqn_sys.eval(
                {removed_var: value}
            )
            dependant_system_copy = dependant_system.eval({removed_var: value})
            dependant_system_copy.add_equation(removed_var, value)
            potential_fixed_points = resolved_partial_eqn_sys.find_all_fixed_points(
                dependant_system_copy
            )
            found_fixed_points.extend(
                (
                    fixed_pt
                    for fixed_pt in potential_fixed_points
                    if int(removed_eqn.eval(fixed_pt)) == value
                )
            )

        return found_fixed_points

    ################################################################################################

    # def as_sympy(self):
    #     variables = tuple(self._equation_dict.keys())
    #
    #     return variables, \
    #            {var: self._equation_dict[var] if is_integer(self._equation_dict[var])
    #            else self._equation_dict[var].as_sympy()
    #            if isinstance(self._equation_dict[var], Expression)
    #            else sympy.Integer(self._equation_dict[var])
    #             for var in self._equation_dict}

    def as_numpy(self) -> Tuple[Tuple[str], Callable]:
        variables: Tuple[str] = tuple(self._equation_dict.keys())

        functions = [
            self._equation_dict[var].as_numpy_str(variables)
            if isinstance(self._equation_dict[var], Expression)
            else str(np.mod(int(self._equation_dict[var]), 3))
            for var in variables
        ]

        function_str = "".join(
            [
                "update_function = lambda state: np.array([",
                ",".join(["np.mod(" + function + ", 3)" for function in functions]),
                "], dtype=np.int64)",
            ]
        )

        # See https://docs.python.org/3/library/functions.html#exec
        locals_dict = dict()
        exec(
            function_str, globals(), locals_dict
        )  # now there is a 'update_function' defined
        update_function = locals_dict["update_function"]
        return variables, update_function

    ################################################################################################

    def as_sbml_qual(self) -> BeautifulSoup:
        soup = BeautifulSoup(features="xml")

        sbml_top_tag: Tag = soup.new_tag(
            "sbml",
            attrs={
                "xmlns": "http://www.sbml.org/sbml/level3/version1/core",
                "level": "3",
                "version": "1",
                "xmlns:qual": "http://www.sbml.org/sbml/level3/version1/qual/version1",
                "qual:required": "true",
            },
        )
        soup.append(sbml_top_tag)

        # not sure if this is part of the spec, but GINsim doesn't work with spaces in the model id
        model = soup.new_tag("model", attrs={"id": "SteadyCellPhenotype_model"})
        sbml_top_tag.append(model)

        species_list = soup.new_tag("qual:listOfQualitativeSpecies")
        model.append(species_list)

        transitions_list = soup.new_tag("qual:listOfTransitions")
        model.append(transitions_list)

        # add species to species list, and transition functions to their own list
        for symbol in self._formula_symbol_table:
            update_formula: ExpressionOrInt = self._equation_dict[symbol]

            # NOTE: I'd prefer if I didn't have to convert the update formula to a polynomial, but
            # the currently version of SBML-qual uses SBML Level 3 revision 1, which doesn't have
            # max or min functions. That means that max/min/not formulas get a lossy conversion
            # when converted to SBML-qual.
            if FORCE_SBML_POLYNOMIAL and isinstance(update_formula, Expression):
                update_formula = update_formula.as_polynomial()

            if (
                isinstance(update_formula, Expression)
                and not update_formula.is_constant()
            ):
                # add to species list
                symbol_tag = soup.new_tag(
                    "qual:qualitativeSpecies",
                    attrs={
                        "qual:maxLevel": 2,
                        "qual:compartment": DEFAULT_COMPARTMENT_NAME,
                        "qual:id": symbol,
                        "qual:constant": "false",
                    },
                )
                species_list.append(symbol_tag)

                transition_tag = soup.new_tag(
                    "qual:transition", attrs={"qual:id": "transition_" + symbol}
                )
                transitions_list.append(transition_tag)

                input_list = soup.new_tag("qual:listOfInputs")
                transition_tag.append(input_list)
                for idx, variable in enumerate(update_formula.get_variable_set()):
                    input_list.append(
                        soup.new_tag(
                            "qual:input",
                            attrs={
                                "qual:qualitativeSpecies": variable,
                                "qual:transitionEffect": "none",
                                "qual:id": f"transition_{symbol}_input_{idx}",
                            },
                        )
                    )

                output_list = soup.new_tag("qual:listOfOutputs")
                transition_tag.append(output_list)
                output_list.append(
                    soup.new_tag(
                        "qual:output",
                        attrs={
                            "qual:qualitativeSpecies": symbol,
                            "qual:transitionEffect": "assignmentLevel",
                            "qual:id": f"transition_{symbol}_output",
                        },
                    )
                )

                function_terms = soup.new_tag("qual:listOfFunctionTerms")
                transition_tag.append(function_terms)

                if SBML_TRUTHTABLE_OUTPUT:
                    if isinstance(update_formula, int):
                        function_terms.append(
                            soup.new_tag(
                                "qual:defaultTerm",
                                attrs={"qual:resultLevel": update_formula},
                            )
                        )
                    else:
                        assert isinstance(update_formula, Expression)
                        truth_table: TruthTable
                        truth_table, counts = update_formula.as_truth_table()

                        if DEBUG_TRUTH_TABLES:
                            with open("debug/" + str(symbol) + ".txt", "w") as file:
                                file.write(
                                    (
                                        ",".join(
                                            [symbol + "(out)"]
                                            + [var for var, val in truth_table[0][0]]
                                        )
                                    )
                                    + "\n"
                                )
                                file.writelines(
                                    [
                                        str(line[1])
                                        + ","
                                        + ",".join(str(val) for var, val in line[0])
                                        + "\n"
                                        for line in truth_table
                                    ]
                                )

                        default_written = False
                        # default = 0
                        if counts[0] > 0:
                            default_written = True
                            function_terms.append(
                                soup.new_tag(
                                    "qual:defaultTerm", attrs={"qual:resultLevel": 0}
                                )
                            )

                        def make_and_term(values: Tuple[Tuple[str, int], ...]) -> Tag:
                            apply = soup.new_tag("apply")
                            apply.append(
                                Tag(
                                    name="and",
                                    is_xml=True,
                                    can_be_empty_element=True,
                                )
                            )
                            for var, val in values:
                                sub_apply = soup.new_tag("apply")
                                sub_apply.append(
                                    Tag(
                                        name="eq",
                                        is_xml=True,
                                        can_be_empty_element=True,
                                    )
                                )
                                sub_apply.append(inner_mathml_variable(var))
                                sub_apply.append(inner_mathml_constant(val))
                                apply.append(sub_apply)
                            return apply

                        # level = 1
                        level1_apply = None
                        if not default_written and counts[1] > 0:
                            default_written = True
                            function_terms.append(
                                soup.new_tag(
                                    "qual:defaultTerm", attrs={"qual:resultLevel": 1}
                                )
                            )
                        elif counts[1] > 0:
                            level1_function_term = soup.new_tag(
                                "qual:functionTerm", attrs={"qual:resultLevel": 1}
                            )
                            function_terms.append(level1_function_term)
                            level1_mathml = soup.new_tag(
                                "math",
                                attrs={"xmlns": "http://www.w3.org/1998/Math/MathML"},
                            )
                            level1_function_term.append(level1_mathml)
                            level1_apply = soup.new_tag("apply")
                            level1_mathml.append(level1_apply)
                            level1_apply.append(
                                Tag(name="or", is_xml=True, can_be_empty_element=True)
                            )

                        # level = 2
                        level2_apply = None
                        if not default_written:
                            function_terms.append(
                                soup.new_tag(
                                    "qual:defaultTerm", attrs={"qual:resultLevel": 2}
                                )
                            )
                        elif counts[2] > 0:
                            level2_function_term = soup.new_tag(
                                "qual:functionTerm", attrs={"qual:resultLevel": 2}
                            )
                            function_terms.append(level2_function_term)
                            level2_mathml = soup.new_tag(
                                "math",
                                attrs={"xmlns": "http://www.w3.org/1998/Math/MathML"},
                            )
                            level2_function_term.append(level2_mathml)
                            level2_apply = soup.new_tag("apply")
                            level2_mathml.append(level2_apply)
                            level2_apply.append(
                                Tag(name="or", is_xml=True, can_be_empty_element=True)
                            )

                        # fill in terms
                        for inputs, value in truth_table:
                            if value == 1 and level1_apply is not None:
                                level1_apply.append(make_and_term(inputs))
                            elif value == 2 and level2_apply is not None:
                                level2_apply.append(make_and_term(inputs))

                else:
                    # here we try to keep the SBML output kind-of like the original functions
                    function_terms.append(
                        soup.new_tag("qual:defaultTerm", attrs={"qual:resultLevel": 0})
                    )

                    level1_function_term = soup.new_tag(
                        "qual:functionTerm", attrs={"qual:resultLevel": 1}
                    )
                    level1_function_term.append(update_formula.as_sbml_qual_relation(1))
                    function_terms.append(level1_function_term)

                    level2_function_term = soup.new_tag(
                        "qual:functionTerm", attrs={"qual:resultLevel": 2}
                    )
                    level2_function_term.append(update_formula.as_sbml_qual_relation(2))
                    function_terms.append(level2_function_term)

            else:
                # set constant and initial level flags
                update_formula = int(update_formula)
                symbol_tag = soup.new_tag(
                    "qual:qualitativeSpecies",
                    attrs={
                        "qual:maxLevel": 2,
                        "qual:compartment": DEFAULT_COMPARTMENT_NAME,
                        "qual:id": symbol,
                        "qual:constant": "true",
                        "qual:initialLevel": str(update_formula),
                    },
                )
                species_list.append(symbol_tag)

        compartments_list = soup.new_tag("listOfCompartments")
        model.append(compartments_list)
        compartments_list.append(
            soup.new_tag(
                "compartment",
                attrs={"constant": "true", "id": DEFAULT_COMPARTMENT_NAME},
            )
        )

        return soup

    ################################################################################################

    def __getitem__(self, item):
        return self._equation_dict[item]

    ################################################################################################

    def continuous_polynomial_system(
        self, continuous_vars: Sequence[str] = None
    ) -> EquationSystem:
        """
        Get continuous version of system

        Parameters
        ----------
        continuous_vars sequence of variable names. if not specified, all are made continuous

        Returns
        -------
        EquationSystem
        """
        if continuous_vars is None:
            continuous_vars = list(self._equation_dict.keys())

        if PARALLEL:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                continuous_equations = dict(
                    pool.starmap(
                        continuity_helper,
                        [
                            (var, eqn, continuous_vars)
                            for var, eqn in self._equation_dict.items()
                        ],
                    )
                )
        else:
            continuous_equations = {
                control_variable: continuity_helper(
                    control_variable, equation, continuous_vars
                )
                for control_variable, equation in self._equation_dict.items()
            }
        return EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=self._formula_symbol_table,
            equation_dict=continuous_equations,
            lines=self._lines,
        )

    ################################################################################################

    def continuous_functional_system(
        self, continuous_vars: Sequence[str] = None
    ) -> EquationSystem:
        """
        Get continuous version of system

        Parameters
        ----------
        continuous_vars sequence of variable names. if not specified, all are made continuous

        Returns
        -------
        EquationSystem
        """
        if continuous_vars is None:
            continuous_vars = set(self._equation_dict.keys())

        continuous_vars = set(continuous_vars)

        continuous_equations = {
            var: Function("CONT", [Monomial.as_var(var), expr])
            if var in continuous_vars
            else expr
            for var, expr in self._equation_dict.items()
        }

        return EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=self._formula_symbol_table,
            equation_dict=continuous_equations,
            lines=self._lines,
        )

    ################################################################################################

    def simplify(self) -> EquationSystem:
        """
        Create simplified system which propagates constant variables to formulas where used

        Returns
        -------
        EquationSystem
        """

        constant_variables = self.constant_variables()
        constant_dict: Dict[str, int] = dict()
        for constant_variable in constant_variables:
            constant = int(self._equation_dict[constant_variable])
            constant_dict[constant_variable] = constant

        equations = {
            var: eqn.eval(constant_dict)
            if isinstance(eqn, Expression)
            else (eqn % 3)
            if isinstance(eqn, int)
            else eqn
            for var, eqn in self._equation_dict.items()
        }

        return EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=self._formula_symbol_table,
            equation_dict=equations,
            lines=self._lines,
        )

    ################################################################################################

    def knockout_system(self, knockouts: Dict[str, int]) -> EquationSystem:
        """
        Create 'knockout' system, where equations for certain variables are set to constants

        Parameters
        ----------
        knockouts: Dict[str, int] format is variable-name: value

        Returns
        -------
        EquationSystem
        """
        equation_dict = deepcopy(self._equation_dict)
        for var, val in knockouts.items():
            assert var in equation_dict, "Can't knockout a variable which isn't present"
            equation_dict[var] = int(val) % 3

        return EquationSystem(
            is_dependant=self._is_dependant,
            formula_symbol_table=self._formula_symbol_table,
            equation_dict=equation_dict,
            lines=self._lines,
        )

    ################################################################################################

    def compose(self, other: EquationSystem) -> EquationSystem:
        assert set(self._equation_dict.keys()) == set(
            other._equation_dict.keys()
        ), "incompatible systems!"

        if PARALLEL:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                # start mapping via eval
                evaluator = partial(
                    evaluate_parallel_helper, mapping_dict=other._equation_dict
                )
                composed_dict = dict(
                    pool.map(evaluator, tuple(self._equation_dict.items()))
                )
        else:
            composed_dict = {
                target_var: eqn.eval(other._equation_dict)
                if isinstance(eqn, Expression)
                else (eqn % 3)
                if isinstance(eqn, int)
                else eqn
                for target_var, eqn in self._equation_dict
            }

        return EquationSystem(
            is_dependant=self._is_dependant and other._is_dependant,
            formula_symbol_table=deepcopy(self._formula_symbol_table),
            equation_dict=composed_dict,
        )

    def self_compose(self, count: int) -> EquationSystem:
        assert count >= 0, "negative powers unsupported!"

        if count == 0:
            # every variable maps to itself in the identity map.
            return EquationSystem(
                is_dependant=self._is_dependant,
                formula_symbol_table=self._formula_symbol_table,
                equation_dict={
                    var: Monomial.as_var(var) for var in self._equation_dict.keys()
                },
            )
        elif count == 1:
            return self
        elif count % 2 == 0:
            square_root_system = self.self_compose(count // 2)
            return square_root_system.compose(square_root_system)
        else:  # count % 2 == 1
            pseudo_square_root_system = self.self_compose(count // 2)
            return pseudo_square_root_system.compose(pseudo_square_root_system).compose(
                self
            )

    ################################################################################################

    def __str__(self):
        if len(self._equation_dict) == 0:
            return "Empty System"
        elif self._is_dependant:
            return "\n".join(
                str(var) + "=" + str(eqn) for var, eqn in self._equation_dict.items()
            )
        else:
            return "\n".join(
                [
                    (
                        str(var)
                        + "="
                        + str(self._equation_dict[var])
                        + (" # " + escape(comment) if comment is not None else "")
                    )
                    if var is not None
                    else ("# " + escape(comment) if comment is not None else "")
                    for var, comment in self._lines
                ]
            )

    __repr__ = __str__

    ################################################################################################

    def __iter__(self):
        return self._equation_dict.items().__iter__()

    ################################################################################################

    def polynomial_output(self, translate_symbol_names=True) -> str:
        if PARALLEL:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                equations_as_polys = list(
                    pool.map(
                        polynomial_output_parallel_helper, self._equation_dict.values()
                    )
                )
        else:
            equations_as_polys = [
                equation.as_polynomial()
                if isinstance(equation, Expression)
                else equation
                for equation in self._equation_dict.values()
            ]

        if translate_symbol_names:
            translation_names = {
                var: "x" + str(idx) for idx, var in enumerate(self.target_variables())
            }
            return "\n".join(
                [
                    "f"
                    + str(idx)
                    + "="
                    + (
                        str(eqn.rename_variables(translation_names))
                        if isinstance(eqn, Expression)
                        else str(eqn)
                    )
                    for idx, eqn in enumerate(equations_as_polys)
                ]
            )
        else:
            return "\n".join(
                [
                    str(target) + "=" + str(eqn)
                    for target, eqn in zip(
                        self._equation_dict.keys(), equations_as_polys
                    )
                ]
            )

    ################################################################################################

    def parse_and_add_equation(self, line: str, line_number: Optional[int] = None):
        """
        Parse a line of the form 'symbol=formula' and add it to the system.

        Parameters
        ----------
        line: str
            A line of the form 'symbol=formula # comment' or pure comment starting with '#'
        line_number: Union[int, None]
            line number in the model, used for error reporting

        Returns
        -------
        None
        """
        line = line.strip()

        comment_idx = line.find("#")
        comment: Optional[str] = None
        if comment_idx != -1:
            comment = line[comment_idx + 1 :].strip()
            line = line[:comment_idx].strip()

        if len(line) == 0:
            self._lines.append((None, comment))
            return  # empty or comment

        if len(line.splitlines()) != 1:
            raise ParseError("This function does not handle multi-line equations.")

        line = line.replace("-", "+(-1)*")  # parsing aide

        tokenized_list = tokenize(line)

        # Formulas should be of the form 'symbol=function'
        if (
            len(tokenized_list) < 3
            or tokenized_list[0][0] != "SYMBOL"
            or tokenized_list[1][0] != "EQUALS"
        ):
            if line_number is not None:
                raise ParseError(
                    f"On line number {line_number}, formula did not"
                    f" begin with a symbol then equals sign!"  # + repr(tokenized_list)
                )
            else:
                raise ParseError(
                    "Formula did not begin with a symbol then equals sign!"
                )
        target_variable = tokenized_list[0][1]

        # the rest should correspond to the formula
        tokenized_formula = tokenized_list[2:]

        # gather the symbols from the function, which may not all be new
        symbols = [
            symbol for (type_str, symbol) in tokenized_formula if type_str == "SYMBOL"
        ]

        equation = translate_to_expression(tokenized_formula)

        # if no errors, make updates
        for symb in symbols:
            if symb not in self._formula_symbol_table:
                self._formula_symbol_table.append(symb)
        self._equation_dict[target_variable] = equation

        self._lines.append((target_variable, comment))

    ################################################################################################
