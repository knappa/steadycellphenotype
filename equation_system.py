from __future__ import annotations

import functools
import itertools
from copy import deepcopy
from typing import Tuple, Sequence, List

from poly import *

UNIVARIATE_FUNCTIONS = ['NOT']
BIVARIATE_FUNCTIONS = ['MAX', 'MIN', 'CONT']


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
    whitespace = [' ', '\t']
    punctuation = ['(', ')', ',', '=', '+', '-', '*', '^']
    function_names = ['MAX', 'MIN', 'NOT']
    while len(input_string) > 0:
        if input_string[0] in whitespace:
            # remove whitespace: spaces and tabs
            input_string = input_string[1:]
        elif input_string[0] == '(':
            tokenized_list.append(('OPEN_PAREN', '('))
            input_string = input_string[1:]
        elif input_string[0] == ')':
            tokenized_list.append(('CLOSE_PAREN', ')'))
            input_string = input_string[1:]
        elif input_string[0] == ',':
            tokenized_list.append(('COMMA', ','))
            input_string = input_string[1:]
        elif input_string[0] == '=':
            tokenized_list.append(('EQUALS', '='))
            input_string = input_string[1:]
        elif input_string[0] == '+':
            tokenized_list.append(('PLUS', '+'))
            input_string = input_string[1:]
        elif input_string[0] == '-':
            tokenized_list.append(('MINUS', '-'))
            input_string = input_string[1:]
        elif input_string[0] == '*':
            tokenized_list.append(('TIMES', '*'))
            input_string = input_string[1:]
        elif input_string[0] == '^':
            tokenized_list.append(('EXP', '^'))
            input_string = input_string[1:]
        elif input_string[0:3] in function_names:
            tokenized_list.append(('FUNCTION', input_string[0:3]))
            input_string = input_string[3:]
        else:
            # must be a name or constant. can be of variable length, terminated by punctuation or
            # whitespace
            index = 0
            while index < len(input_string) and \
                    not input_string[index] in punctuation and \
                    not input_string[index] in whitespace:
                index += 1
            if index > 0:
                try:
                    # check to see if this is a constant.
                    const = int(input_string[0:index])
                    tokenized_list.append(('CONSTANT', const))
                except ValueError:
                    # if it isn't parsable as an int, it is a symbol
                    tokenized_list.append(('SYMBOL', input_string[0:index]))
                input_string = input_string[index:]
            else:
                raise Exception('Error in tokenization, cannot understand what this is')
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

def translate_to_expression(formula: Sequence[Tuple[str, Union[str, int]]]) -> Union[Expression, int]:
    """
    recursive descent parser for formula translation
    recurses left-to-right, building the formula on the way back up

    :param formula: tokenized list
    :return: Expression
    """
    if len(formula) <= 0:
        raise ParseError("can\'t parse an empty formula")

    next_token_type, next_token_content = formula[0]

    # try to parse addition and subtraction
    for additive_index in find_all(['PLUS', 'MINUS'], formula):
        try:
            argument_one = translate_to_expression(formula[0:additive_index])
            argument_two = translate_to_expression(formula[additive_index + 1:])

            # we succeed!
            if formula[additive_index][0] == 'PLUS':
                return argument_one + argument_two
                # return BinaryOperation('PLUS', argument_one, argument_two)
            else:
                assert formula[additive_index][0] == 'MINUS'
                return argument_one - argument_two
                # return BinaryOperation('MINUS', argument_one, argument_two)
        except ParseError:
            pass

    # try to parse as a unary minus: -ARG
    if next_token_type == 'MINUS':
        try:
            argument = translate_to_expression(formula[1:])
            # we succeed!
            return -argument
            # return UnaryRelation('MINUS', argument)
        except ParseError:
            pass

    # try to parse multiplication
    for mult_index in find_all('TIMES', formula):
        try:
            argument_one = translate_to_expression(formula[0:mult_index])
            argument_two = translate_to_expression(formula[mult_index + 1:])
            # we succeed!
            return argument_one * argument_two
            # return BinaryOperation('TIMES', argument_one, argument_two)
        except ParseError:
            pass

    # try to parse exponentiation --- we only allow atomic integer constants in the exponent because
    # these are not to be taken mod 3
    if len(formula) >= 3 and formula[-2][0] == 'EXP' and formula[-1][0] == 'CONSTANT':
        exp_index = len(formula) - 2
        try:
            base = translate_to_expression(formula[0:exp_index])
            # again, we only accept constants in the exponent!
            if formula[-1][0] != 'CONSTANT':
                raise ParseError("must have constant in the exponent")
            exponent = formula[-1][1]
            # we succeed!
            return base ** exponent
            # return BinaryOperation('EXP', base, exponent)
        except ParseError:
            pass

    # try to parse parentheses: (ARG)
    if next_token_type == 'OPEN_PAREN' and formula[-1][0] == 'CLOSE_PAREN':
        try:
            argument = translate_to_expression(formula[1:-1])
            # we succeed!
            return argument
        except ParseError:
            pass

    # try to parse CONST/SYMB
    if len(formula) == 1:
        if next_token_type == 'CONSTANT':
            constant: int = next_token_content
            return constant
        elif next_token_type == 'SYMBOL':
            symbol_text = next_token_content
            return Monomial({symbol_text: 1})
        else:
            raise ParseError("Invalid singleton")

    # try to parse functions
    if next_token_type == 'FUNCTION':
        function_name = next_token_content
        # formatting sanity check; must have FUNCTION OPEN_PAREN ... CLOSE_PAREN
        if len(formula) < 2 or formula[1][0] != 'OPEN_PAREN' or formula[-1][0] != 'CLOSE_PAREN':
            raise ParseError('Parse error, must have form FUNCTION OPEN_PAREN ... CLOSE_PAREN')

        # lop off the function name and parentheses
        inner_formula = formula[2:-1]

        if function_name in UNIVARIATE_FUNCTIONS:
            argument = translate_to_expression(inner_formula)  # may throw ParseError, which we do not catch here
            # we succeed!
            return Function(function_name, [argument])
        elif function_name in BIVARIATE_FUNCTIONS:
            # try to find the comma
            for comma_index in find_all('COMMA', inner_formula):
                try:
                    argument_one = translate_to_expression(inner_formula[0:comma_index])
                    argument_two = translate_to_expression(inner_formula[comma_index + 1:])
                    # if we succeed, return the function
                    return Function(function_name, [argument_one, argument_two])
                except ParseError:
                    # try another comma
                    pass
            # no comma gave a valid parse
            raise ParseError('could not parse function call')
        else:
            # function name was unrecognized
            raise ParseError('This is not a function I recognize!')

    # in the case that nothing worked
    raise ParseError('Parse error!')


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


def continuity_parallel_helper(control_variable, equation: Expression, continuous_vars):
    """
    Because python won't let you use lambdas in multiprocessing
    :param control_variable:
    :param equation:
    :param continuous_vars:
    :return:
    """
    if control_variable not in continuous_vars:
        return equation
    elif isinstance(equation, int) or equation.is_constant():
        return int(equation)
    else:
        return equation.continuous_polynomial_version(control_variable)


def polynomial_output_parallel_helper(equation: Union[Expression, int]):
    """
    Because python won't let you use lambdas in multiprocessing
    :param equation:
    :return:
    """
    if isinstance(equation, int) or equation.is_constant():
        return int(equation)
    else:
        return equation.as_polynomial()


####################################################################################################

class EquationSystem(object):

    def __init__(self, lines: str = None, formula_symbol_table=None, equations=None, target_variables=None):
        assert lines is None or \
               (formula_symbol_table is None and equations is None and target_variables is None),\
                "Must specify lines _or_ formula_symbol_table, equations, and target_variables"
        if lines is None:
            self._formula_symbol_table = formula_symbol_table if formula_symbol_table is not None else []
            self._equations: List[Union[int, Expression]] = equations if equations is not None else []
            self._target_variables = target_variables if target_variables is not None else []
        else:
            self._formula_symbol_table = []
            self._equations: List[Union[int, Expression]] = []
            self._target_variables = []
            lines = lines.strip()
            for line in lines.splitlines():
                self.parse_and_add_equation(line)

    def as_poly_system(self) -> EquationSystem:
        formula_symbol_table = deepcopy(self._formula_symbol_table)
        equations = [eqn.as_polynomial() if isinstance(eqn, Expression) else eqn for eqn in self._equations]
        target_variables = deepcopy(self._target_variables)

        return EquationSystem(formula_symbol_table=formula_symbol_table,
                              equations=equations,
                              target_variables=target_variables)

    def symbol_table(self):
        return set(self._formula_symbol_table).union(self._target_variables)

    def formula_symbol_table(self):
        return deepcopy(self._formula_symbol_table)

    def target_variables(self):
        return deepcopy(self._target_variables)

    def variables_that_vary(self):
        return [var for var, eqn in zip(self._target_variables, self._equations)
                if type(eqn) != int and not eqn.is_constant()]

    def constant_variables(self):
        return [var for var, eqn in zip(self._target_variables, self._equations)
                if type(eqn) == int or eqn.is_constant()]

    def consistent(self):
        """
        Check if this makes sense as an update function. The set of formula symbols must be contained in the set
        of target variables
        :return: boolean value
        """
        return set(self._formula_symbol_table) <= set(self._target_variables)

    ################################################################################################

    def continuous_system(self, continuous_vars: Sequence[str] = None) -> EquationSystem:
        """
        Get continuous version of system

        :param continuous_vars: sequence of variable names. if not specified, all are made continuous
        except when listed in non_continuous_vars
        :return: EquationSystem
        """
        if continuous_vars is None:
            continuous_vars = self._target_variables
        try:
            import multiprocessing
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                continuous_equations = list(pool.starmap(continuity_parallel_helper,
                                                         zip(self._target_variables,
                                                             self._equations,
                                                             itertools.repeat(continuous_vars))))
                return EquationSystem(formula_symbol_table=self._formula_symbol_table,
                                      equations=continuous_equations,
                                      target_variables=self._target_variables)
        except ImportError:
            continuous_equations = [equation.continuous_polynomial_version(control_variable)
                                    for control_variable, equation in zip(self._target_variables, self._equations)]
            return EquationSystem(formula_symbol_table=self._formula_symbol_table,
                                  equations=continuous_equations,
                                  target_variables=self._target_variables)

    ################################################################################################

    def simplify(self) -> EquationSystem:
        formula_symbol_table = deepcopy(self._formula_symbol_table)
        equations = deepcopy(self._equations)
        target_variables = deepcopy(self._target_variables)

        constant_variables = self.constant_variables()
        constant_dict = dict()
        for constant_variable in constant_variables:
            idx = target_variables.index(constant_variable)
            constant = int(equations[idx])
            constant_dict[constant_variable] = constant

        equations = [eqn.eval(constant_dict) if isinstance(eqn, Expression) else eqn for eqn in equations]

        return EquationSystem(formula_symbol_table=formula_symbol_table,
                              equations=equations,
                              target_variables=target_variables)

    ################################################################################################

    def knockout_system(self, knockouts: Dict[str, int]) -> EquationSystem:
        formula_symbol_table = deepcopy(self._formula_symbol_table)
        equations = deepcopy(self._equations)
        target_variables = deepcopy(self._target_variables)

        for idx, variable in enumerate(target_variables):
            if variable in knockouts:
                equations[idx] = knockouts[variable]

        # TODO: below is a round of simplification, is that appropriate?
        # equations = [eqn.eval(knockouts) if isinstance(eqn, Expression) else eqn for eqn in equations]

        return EquationSystem(formula_symbol_table=formula_symbol_table,
                              equations=equations,
                              target_variables=target_variables)

    ################################################################################################

    def compose(self, other: EquationSystem) -> EquationSystem:
        assert set(self._target_variables) == set(other._target_variables), "incompatible systems!"

        # set up the dict which defines the mapping
        other_dict = {target_var: eqn
                      for target_var, eqn in zip(other._target_variables, other._equations)}

        # NOTE: it seems that Pool is not available on AWS or on Android (I want to run this on my tablet)
        try:
            # noinspection PyUnresolvedReferences
            import multiprocessing
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                # start mapping via eval
                evaluator = functools.partial(evaluate_parallel_helper,
                                              mapping_dict=other_dict)
                composed_dict = dict(pool.map(evaluator,
                                              tuple(zip(self._target_variables, self._equations))))
        except ImportError:
            composed_dict = {target_var: eqn.eval(other_dict)
                             for target_var, eqn in zip(self._target_variables, self._equations)}

        return EquationSystem(formula_symbol_table=deepcopy(self._formula_symbol_table),
                              equations=[composed_dict[var] for var in self._target_variables],
                              target_variables=deepcopy(self._target_variables))

    def self_compose(self, count: int) -> EquationSystem:
        assert count >= 0, "negative powers unsupported!"

        if count == 0:
            # every variable maps to itself in the identity map.
            return EquationSystem(formula_symbol_table=self._formula_symbol_table,
                                  equations=[Monomial.as_var(var) for var in self._target_variables],
                                  target_variables=self._target_variables)
        elif count == 1:
            return self
        elif count % 2 == 0:
            square_root_system = self.self_compose(count // 2)
            return square_root_system.compose(square_root_system)
        else:  # count % 2 == 1
            pseudo_square_root_system = self.self_compose(count // 2)
            return pseudo_square_root_system.compose(pseudo_square_root_system).compose(self)

    ################################################################################################

    def __str__(self):
        if len(self._equations) == 0:
            return "Empty System"
        else:
            return "\n".join([str(var) + "=" + str(eqn)
                              for var, eqn in zip(self._target_variables, self._equations)])

    __repr__ = __str__

    ################################################################################################

    def __iter__(self):
        class EquationSystemIterator(object):
            def __init__(self, targets, equations):
                self._target_varaibles = tuple(targets)
                self._equations = tuple(equations)
                self._count = 0

            def __next__(self):
                if self._count < len(self._target_varaibles):
                    count = self._count
                    self._count += 1
                    return self._target_varaibles[count], self._equations[count]
                else:
                    raise StopIteration()

        return EquationSystemIterator(self._target_variables, self._equations)

    ################################################################################################

    def polynomial_output(self, translate_symbol_names=True) -> str:
        try:
            # noinspection PyUnresolvedReferences
            import multiprocessing
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                equations_as_polys = list(pool.map(polynomial_output_parallel_helper, self._equations))
        except ImportError:
            equations_as_polys = [equation.as_polynomial() if isinstance(equation, Expression) else equation
                                  for equation in self._equations]

        if translate_symbol_names:
            translation_names = {var: 'x' + str(idx) for idx, var in enumerate(self.target_variables())}
            return "\n".join(
                ['f' + str(idx) + '=' + (str(eqn.rename_variables(translation_names))
                                         if isinstance(eqn, Expression) else str(eqn))
                 for idx, eqn in enumerate(equations_as_polys)])
        else:
            return "\n".join(
                [str(target) + '=' + str(eqn)
                 for target, eqn in zip(self._target_variables, equations_as_polys)])

    ################################################################################################

    def parse_and_add_equation(self, line: str):
        """
        Parse a line of the form 'symbol=formula' and add it to the system
        :param line: a string
        :return: None
        """
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            return  # empty or comment

        if len(line.splitlines()) != 1:
            raise ParseError("This function does not handle multi-line equations.")

        tokenized_list = tokenize(line)

        # Formulas should be of the form 'symbol=function'
        if len(tokenized_list) < 3 or tokenized_list[0][0] != 'SYMBOL' or tokenized_list[1][0] != 'EQUALS':
            raise ParseError('Formula did not begin with a symbol then equals sign!')
        target_variable = tokenized_list[0][1]

        # the rest should correspond to the formula
        tokenized_formula = tokenized_list[2:]

        # gather the symbols from the function, which may not all be new
        symbols = [symbol for (type_str, symbol) in tokenized_formula
                   if type_str == 'SYMBOL']

        equation = translate_to_expression(tokenized_formula)

        # if no errors, make updates
        for symb in symbols:
            if symb not in self._formula_symbol_table:
                self._formula_symbol_table.append(symb)
        self._equations.append(equation)
        self._target_variables.append(target_variable)

    ################################################################################################
