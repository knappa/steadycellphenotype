from __future__ import annotations

import operator
from copy import deepcopy
from functools import partial, reduce
from html import escape
# noinspection PyUnresolvedReferences
from math import floor
from typing import Callable, Optional, Sequence

# noinspection PyProtectedMember
from bs4 import ResultSet

from steady_cell_phenotype.poly import *

UNIVARIATE_FUNCTIONS = ["NOT"]
BIVARIATE_FUNCTIONS = ["MAX", "MIN", "CONT"]

DEFAULT_COMPARTMENT_NAME = "compartment1"

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
        elif input_string[0:3].upper() in function_names:
            tokenized_list.append(("FUNCTION", input_string[0:3].upper()))
            input_string = input_string[3:]
        else:
            # must be a name or constant. can be of variable length, terminated by punctuation or
            # whitespace
            index = 0
            while (
                index < len(input_string)
                and not input_string[index] in punctuation
                and not input_string[index] in whitespace
            ):
                index += 1
            if index > 0:
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
        return variable_name, expression.eval(mapping_dict)


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


def parse_mathml_to_function(
    input_variables: List[str], mathml_tag: Tag
) -> Callable[..., bool]:
    def parse_mathml_helper(input_variable_list: List[str], mathml_subtag) -> Callable:
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
        if tag_type == "cn":
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
        if tag_type == "ci":
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
        if tag_type == "apply":
            tag_contents: List[Tag] = list(
                filter(lambda x: type(x) == Tag, mathml_subtag.contents)
            )
            if len(tag_contents) <= 0:
                raise ParseError("Empty apply")
            operation = tag_contents[0].name
            operand_functions: List[Callable] = [
                parse_mathml_helper(input_variables, operand_tag)
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
                    return operator_function(
                        operand_a(input_values), operand_b(input_values)
                    )

                return f

            # we didn't handle the parse
            raise ParseError(
                f"Unsupported function: {operation} on {len(operand_functions)} operands"
            )

    # examine the contents of the mathml, there may be strings (probably \n) which we should ignore
    contents: List[Tag] = list(filter(lambda x: type(x) == Tag, mathml_tag.contents))
    if len(contents) != 1:
        raise ParseError

    return parse_mathml_helper(input_variables, contents[0])


def parse_sbml_qual_function(
    input_variables: List[str], function_terms: Tag
) -> Callable[..., int]:
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
        for value, function in conditional_levels:
            if function(input_values):
                return value
        return default_level

    return function_realization


####################################################################################################


@attrs(init=False, slots=True, repr=False, str=False)
class EquationSystem(object):
    _formula_symbol_table: List[str] = attrib()
    _equation_dict: Dict[str, ExpressionOrInt] = attrib()
    _lines: List[Tuple[Optional[str], Optional[str]]] = attrib()

    # _lines: Tuples of variable name and comment, both optional, for printing

    def __init__(
        self,
        *,
        formula_symbol_table: List[str] = None,
        equation_dict: Dict[str, ExpressionOrInt] = None,
        lines: List[Tuple[Optional[str], Optional[str]]] = None,
    ):
        if formula_symbol_table is not None and equation_dict is not None:
            self._formula_symbol_table = deepcopy(formula_symbol_table)
            self._equation_dict = deepcopy(equation_dict)
            if lines is None:
                self._lines = [(symbol, None) for symbol in formula_symbol_table]
            else:
                self._lines = deepcopy(lines)
        elif formula_symbol_table is None and equation_dict is None:
            self._formula_symbol_table = []
            self._equation_dict = dict()
            self._lines = list()
        else:
            raise RuntimeError(
                "Must specify either an empty system,"
                " or both formula_symbol_table and equation_dict"
            )

    @staticmethod
    def from_text(lines: str) -> EquationSystem:
        """
        Create a system of equations from text

        Returns
        -------
        EquationSystem
        """
        equation_system: EquationSystem = EquationSystem()
        for line_number, line in enumerate(lines.strip().splitlines()):
            equation_system.parse_and_add_equation(line)
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
        for species in species_list:
            if "qual:maxlevel" not in species.attrs:
                return (
                    f"No max level provided for {species.attrs['qual:id']}, expecting 2"
                )

            try:
                if int(species.attrs["qual:maxlevel"]) != 2:
                    return (
                        f"Max level provided for {species.attrs['qual:id']}"
                        f" is {species.attrs['qual:maxlevel']}, expecting 2"
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
                if "qual:initiallevel" in species.attrs:
                    # if the initial level is set then we make the variable constant in the
                    # sense that the update function sets that particular value.
                    try:
                        constant_level = int(species.attrs["qual:initiallevel"])
                        equations[species.attrs["qual:id"]] = constant_level
                    except ValueError:
                        return (
                            f"Level provided for {species.attrs['qual:id']} is"
                            f" {species.attrs['qual:initiallevel']}, cannot parse as an integer"
                        )

                else:
                    # if the initial level isn't set then we make the variable constant in the
                    # sense that the update function is the identity.
                    equations[species.attrs["qual:id"]] = Monomial.as_var(
                        species.attrs["qual:id"]
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
                    input_variables, function_terms[0]
                )
            except ParseError as e:
                return "Could not parse functions " + str(e)

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
            formula_symbol_table=symbols, equation_dict=equations, lines=lines
        )

    def as_poly_system(self) -> EquationSystem:
        formula_symbol_table = deepcopy(self._formula_symbol_table)
        equation_dict = {
            target_var: eqn.as_polynomial() if isinstance(eqn, Expression) else eqn
            for target_var, eqn in self._equation_dict.items()
        }

        return EquationSystem(
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

    def eval(self, state: Dict[str, int]) -> Dict[str, int]:
        if state.keys() != set(self._equation_dict.keys()):
            raise RuntimeError("Evaluating state on incorrect set of variables")
        return {
            variable: formula.eval(state)
            if isinstance(formula, Expression)
            else formula
            for variable, formula in self._equation_dict.items()
        }

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

        model = soup.new_tag("model", attrs={"id": "SteadyCellPhenotype model"})
        sbml_top_tag.append(model)

        species_list = soup.new_tag(
            "qual:listOfQualitativeSpecies",
            attrs={
                "xmlns:qual": "http://www.sbml.org/sbml/level3/version1/qual/version1"
            },
        )
        model.append(species_list)

        transitions_list = soup.new_tag(
            "qual:listOfTransitions",
            attrs={
                "xmlns:qual": "http://www.sbml.org/sbml/level3/version1/qual/version1"
            },
        )
        model.append(transitions_list)

        # add species to species list, and transition functions to their own list
        for symbol in self._formula_symbol_table:
            update_formula: ExpressionOrInt = self._equation_dict[symbol]

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
            var: eqn.eval(constant_dict) if isinstance(eqn, Expression) else eqn
            for var, eqn in self._equation_dict.items()
        }

        return EquationSystem(
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
                for target_var, eqn in self._equation_dict
            }

        return EquationSystem(
            formula_symbol_table=deepcopy(self._formula_symbol_table),
            equation_dict=composed_dict,
        )

    def self_compose(self, count: int) -> EquationSystem:
        assert count >= 0, "negative powers unsupported!"

        if count == 0:
            # every variable maps to itself in the identity map.
            return EquationSystem(
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

    def parse_and_add_equation(self, line: str):
        """
        Parse a line of the form 'symbol=formula' and add it to the system.

        Parameters
        ----------
        line: str
            A line of the form 'symbol=formula # comment' or pure comment starting with '#'

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
            raise ParseError("Formula did not begin with a symbol then equals sign!")
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
