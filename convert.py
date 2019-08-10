#!/usr/bin/env python3
# Copyright Adam C. Knapp 2018
# License: CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/
#
# A cross-compiler, from a 3 state dynamical system written in MAX/MIN/NOT formulae to its canonical
# mod-3 polynomial representation.

import argparse
import functools
import os
from multiprocessing import Pool

from poly import *


####################################################################################################

def tokenize(input_string: str) -> list:
    """This tokenizer takes a (possibly multiline) string and tokenizes it into a list of 2-tuples. Each
       tuple is of the form (TOKEN_TYPE, VALUE) where TOKEN_TYPE can be one of OPEN_PAREN,
       CLOSE_PAREN, COMMA, EQUALS, PLUS, MINUS, TIMES, EXP, FUNCTION, or SYMBOL. VALUE only contains
       non-redundant information in the case of SYMBOL, in which is contains the string
       representation of the symbol.

       This method can fail for invalid equations, throwing an exception.

    """
    tokenized_list = []
    whitespace = [' ', '\t', '\n', '\r']
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

def generate_symbol_table(tokenized_list: list, translate_to_xs=True):
    """generates a symbol table from a tokenized list. if a symbol does not occur _exactly_ once in the
       form SYMBOL=FORMULA, an exception is thrown

       returns a tuple of the form (num_of_symbols, symbol_table)
    """
    symbol_table = dict()
    # first, find everything of the form 'SYMBOL=' and enter them into table
    counter = 0
    for index in range(len(tokenized_list) - 1):
        if tokenized_list[index][0] == 'SYMBOL' and tokenized_list[index + 1][0] == 'EQUALS':
            symbol = tokenized_list[index][1]
            if symbol in tokenized_list:
                raise Exception('%s occurs at least twice in the form %s='%(symbol, symbol))
            symbol_table[symbol] = counter
            symbol_table[counter] = symbol
            counter += 1

    # check that every symbol occurs in the above way
    for (token_type, token) in tokenized_list:
        if token_type == 'SYMBOL' and token not in symbol_table:
            raise Exception('Found symbol %s which does not appear in the form %s=...'%(token, token))

    # if we should translate symbol names to x0, x1, etc. then we should do that now.
    if translate_to_xs:
        # translate the tokenized list
        tokenized_list = [(symbol_type, text) if symbol_type != 'SYMBOL' else ('SYMBOL', 'x%d'%symbol_table[text])
                          for (symbol_type, text) in tokenized_list]
        # replace the symbol table with the new one
        symbol_table = {n: ('x%d'%n) for n in range(counter)}
        symbol_table.update({('x%d'%n): n for n in range(counter)})
    return counter, symbol_table, tokenized_list


####################################################################################################

def separate_equations(tokenized_list):
    """breaks lists of terms of the form target_symbol=formula into lists of terms of the form
    (target_symbol,formula)"""
    equation_list = []

    while len(tokenized_list) > 0:
        if tokenized_list[0][0] != 'SYMBOL':
            raise Exception('Formula did not begin with a symbol!')

        if tokenized_list[1][0] != 'EQUALS':
            raise Exception('No equals sign in formula!')

        # find the next equals sign, if present
        index = 2
        while index < len(tokenized_list) and tokenized_list[index][0] != 'EQUALS':
            index += 1

        # if we didn't run off the end, we found an equals sign and we have to back up a bit to get
        # to point to the symbol before it
        if index != len(tokenized_list):
            index -= 1

        target = tokenized_list[0][1]
        formula = tokenized_list[2:index]
        if len(formula) == 0:
            raise Exception('Empty formula for %s!'%target)
        equation_list.append((target, formula))

        # done with this bit, so junk it
        tokenized_list = tokenized_list[index:]

    return equation_list


####################################################################################################

class ParseError(Exception):
    """Exception raised when we go down a wrong path in parsing

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


####################################################################################################

def find_all(find_token_string, formula):
    """generator which yeilds all indices (in order) of occurences of a particular type
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
# the actual recursive descent parser
####################################################################################################

def translate_to_expression_helper(formula, symbol_table):
    """helper function for formula translation 

       recurses left-to-right, building the formula on the way back up

    """

    if len(formula) <= 0:
        raise ParseError("can\'t parse an empty formula")

    next_token_type, next_token_content = formula[0]

    # try to parse addition and subtraction
    for additive_index in find_all(['PLUS', 'MINUS'], formula):
        try:
            argument_one, remaining_formula = translate_to_expression_helper(formula[0:additive_index],
                                                                             symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("more stuff on left side of + than there should be")

            argument_two, remaining_formula = translate_to_expression_helper(formula[additive_index + 1:],
                                                                             symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("more stuff on right side of + than there should be")

            # we suceed!
            if formula[additive_index][0] == 'PLUS':
                return BinaryOperation('PLUS', argument_one, argument_two), []
            else:
                assert formula[additive_index][0] == 'MINUS'
                return BinaryOperation('MINUS', argument_one, argument_two), []
        except ParseError:
            pass

    # try to parse as a unary minus: -ARG
    if next_token_type == 'MINUS':
        try:
            argument, remaining_formula = translate_to_expression_helper(formula[1:],
                                                                         symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("more stuff on right side of unary - than there should be")
            # we succeed!
            return UnaryRelation('MINUS', argument), []
        except ParseError:
            pass

    # try to parse multiplication
    for mult_index in find_all('TIMES', formula):
        try:
            argument_one, remaining_formula = translate_to_expression_helper(formula[0:mult_index],
                                                                             symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("more stuff on left side of * than there should be")

            argument_two, remaining_formula = translate_to_expression_helper(formula[mult_index + 1:],
                                                                             symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("more stuff on right side of * than there should be")

            # we suceed!
            return BinaryOperation('TIMES', argument_one, argument_two), []
        except ParseError:
            pass

    # try to parse exponentiation --- we only allow atomic integer constants in the exponent because
    # these are not to be taken mod 3
    if len(formula) >= 3 and formula[-2][0] == 'EXP' and formula[-1][0] == 'CONSTANT':
        exp_index = len(formula) - 2
        try:
            base, remaining_formula = translate_to_expression_helper(formula[0:exp_index],
                                                                     symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("more stuff on left side of ^ than there should be")

            # again, we only accept constants in the exponent!
            exponent = formula[-1][1]

            # we suceed!
            return BinaryOperation('EXP', base, exponent), []
        except ParseError:
            pass

    # try to parse parentheses: (ARG)
    if next_token_type == 'OPEN_PAREN' and formula[-1][0] == 'CLOSE_PAREN':
        try:
            argument, remaining_formula = translate_to_expression_helper(formula[1:-1],
                                                                         symbol_table)
            if len(remaining_formula) > 0:
                raise ParseError("the interior of ()'s doesn't parse")
            # we succeed!
            return argument, []
        except ParseError:
            pass

    # try to parse CONST/SYMB/FUNC    

    if next_token_type == 'CONSTANT':
        constant = next_token_content
        return constant*Monomial.unit(), formula[1:]
    elif next_token_type == 'SYMBOL':
        symbol_text = next_token_content
        return Monomial({symbol_text: 1}), formula[1:]
    elif next_token_type == 'FUNCTION':
        function_name = next_token_content
        # formatting sanity check
        if len(formula) < 2 or formula[1][0] != 'OPEN_PAREN':
            raise ParseError('Parse error, no open paren after function name!')

        # lop off the function name and open paren
        formula = formula[2:]

        if function_name == 'NOT':
            # try to find the closing parenthesis
            for close_paren_index in find_all('CLOSE_PAREN', formula):
                try:
                    argument, remaining_formula = translate_to_expression_helper(formula[0:close_paren_index],
                                                                                 symbol_table)
                    if len(remaining_formula) > 0:
                        raise ParseError("more stuff in a not than there should be")
                    else:
                        return Function('NOT', [argument]), formula[close_paren_index + 1:]
                except ParseError:
                    pass

            raise ParseError('could not find the end of the argument to NOT')

        elif function_name == 'MAX' or function_name == 'MIN':
            # try to find the comma
            for comma_index in find_all('COMMA', formula):
                try:
                    argument_one, remaining_formula = translate_to_expression_helper(formula[0:comma_index],
                                                                                     symbol_table)
                    if len(remaining_formula) > 0:
                        raise ParseError("more stuff in an argument than there should be")
                    for close_paren_index in find_all('CLOSE_PAREN', formula):
                        try:
                            argument_two, remaining_formula = translate_to_expression_helper(
                                formula[comma_index + 1:close_paren_index],
                                symbol_table)
                            if len(remaining_formula) > 0:
                                raise ParseError("more stuff in an argument than there should be")
                            else:
                                return Function(function_name, [argument_one, argument_two]), formula[
                                                                                              close_paren_index + 1:]
                        except ParseError:
                            # try another close paren
                            pass
                except ParseError:
                    # try another comma
                    pass
            # no comma + ) gave a valid parse
            raise ParseError('could not parse function call')
        else:
            # function name was neither MAX not MIN
            raise ParseError('This is not a function I recognize, how would we even get here?')
    else:
        # no sensible way to get here
        raise ParseError('Parse error!')


def translate_to_expression(formula, symbol_table):
    """takes a formula in the form of a list fo tokens and returns an object of type Expression"""
    expression, formula_leftovers = translate_to_expression_helper(formula, symbol_table)
    if len(formula_leftovers) > 0:
        raise Exception('Parse error!')

    return expression


def parse(input_string, translate_symbol_names_to_xs):
    """returns a list of (formula_name,formula) pairs. the returned formula is (some subtype of)
       Expression and the formula_name is a str

    """
    tokenized_list = tokenize(input_string)
    num_symbols, symbol_table, tokenized_list = generate_symbol_table(tokenized_list,
                                                                      translate_symbol_names_to_xs)
    equation_list = separate_equations(tokenized_list)
    output_formulae = [(target, translate_to_expression(formula, symbol_table))
                       for target, formula in equation_list]
    return symbol_table, output_formulae


# end of recursive descent parser

####################################################################################################

def formulae_to_str_formulae(formulae, translate_symbol_names_to_xs=True, as_polynomials=True):
    """two options here: can translate the formulae to be f0=... rather than x0=... 

    also, the formula can be represented in MAX/MIN/NOT form or in polynomial form

    """
    return [('f' + target[1:] if translate_symbol_names_to_xs else target) + "=" + \
            str(poly_formula.as_polynomial() if as_polynomials else poly_formula) \
            for target, poly_formula in formulae]


####################################################################################################
# self-composition

def evaluate(pair, mapping_dict):
    """helper function for self_compose, needed to make parallel evaluation work"""
    variable_name, expression = pair
    return variable_name, expression.eval(mapping_dict)


def self_compose(base_formulae, power):
    # every variable maps to itself in the identity map. i.e. zero-th power 
    formulae = {variable_name: Monomial.as_var(variable_name) for variable_name, _ in base_formulae}
    # start mapping via eval
    pool = Pool(4)
    for _ in range(power):
        # set up the dict which defines the mapping
        mapping_dict = {variable_name: formula for variable_name, formula in formulae.items()}
        evaluator = functools.partial(evaluate, mapping_dict=mapping_dict)
        formulae = dict(pool.map(evaluator, base_formulae))
        # formulae = { variable_name: function.eval(mapping_dict) for variable_name, function in base_formulae }
    return [(variable_name, formulae[variable_name]) for variable_name, _ in base_formulae]


####################################################################################################
#
# the following methods convert a system of polynomials into one which is "continuous" in the sense
# that application of the system does not change the per-coordinate values by more than 1. This is
# accomplished by a type of curve fitting. Fortunately, the formula for this
#
# g(x) = sum_{c\in \F_3^n} h(c) prod_{j=0}^n (1-(x_j-c_j)**2)
#
# (as seen in the PLoS article, doi:10.1371/journal.pcbi.1005352.t003 pg 16/24) admits a recursive
# formulation. That is, for a polynomial x_k = f_k(x_0,x_1,...,x_l) we can select one of the
# variables, say x_0 and reduce the polynomial each of 3-ways x_0=0, x_0=1, and x_0=2. This
# correspondingly divides the sum into those which have each of the 3 types of terms
# (1-(x_0-c_0)**2) for c_0=0, c_0=1, and c_0=2
#
# fortunately, (1-(x_j-0)**2)+(1-(x_j-1)**2)+(1-(x_j-2)**2) = 1 so if the evaluations of f become
# constant or even simply eliminate a variable, we need no longer consider that variable.
#
# recursion proceeds by eliminating variables in this manner, multiplying by the appropriate fitting
# term (1-(x_j-c_j)**2) (c_j being the evaluated value of x_j) on the way up.
#
# this comment is not really the place for a full proof of this method, but the proof is easily
# obtained from the above. 
#
####################################################################################################

def get_continuous_formulae(base_formulae, omitted_vars):
    """convert formulae to continuous versions"""
    # the formula for variable "var" is made continuous, first in "var", then recursively in the
    # other variables in the formula
    pool = Pool(8)
    base_formulae_with_omitted_vars = [(var,formula, omitted_vars) for var, formula in base_formulae]
    continuous_versions = list(pool.map(unpacker, base_formulae_with_omitted_vars))
    return continuous_versions
    # return [ get_continuous_per_formula(var, formula) for var, formula in base_formulae ]


def unpacker(three_tuple):
    """pool.map wants to send 3-tuples as a single item to get_continuous_per_formula, that's kind of
    dumb and I don't see an obvious way to unpack them without this dumb little function. So here it is

    """
    var, formula, omitted_vars = three_tuple
    if omitted_vars is None or var not in omitted_vars:
        return get_continuous_per_formula(var, formula)
    else:
        return var, formula


def h(x, fx):
    """helper function as in the PLoS article, doi:10.1371/journal.pcbi.1005352.t003 pg 16/24"""
    if fx > x:
        return x + 1
    elif fx < x:
        return x - 1
    else:
        return x


def get_continuous_per_formula(var, formula):
    # take the easy out
    if formula.is_constant():
        return var, formula
    # go through the whole buisness for the target variable, first
    accumulator = Mod3Poly.zero()
    for base_value in range(3):
        evaluated_poly = formula.eval({var: base_value})
        if evaluated_poly.is_constant():
            computed_value = int(evaluated_poly)
            continuous_value = h(base_value, computed_value)
            accumulator += continuous_value*(1 - (Monomial.as_var(var) - base_value) ** 2)
        else:
            accumulator += get_continuous_per_formula_helper(base_value, evaluated_poly)* \
                           (1 - (Monomial.as_var(var) - base_value) ** 2)
    return var, accumulator


def get_continuous_per_formula_helper(base_var_val, formula):
    """helper function for get_continuous_formulae, gets continuous version of the formula for variable
    var

    """
    # find some variable
    var = tuple(formula.get_var_set())[0]

    # iterate over the ways of setting that variable: 0, 1, 2
    accumulator = Mod3Poly.zero()
    for value in range(3):
        evaluated_poly = formula.eval({var: value})
        if type(evaluated_poly) == int or evaluated_poly.is_constant():
            computed_value = int(evaluated_poly)
            continuous_value = h(base_var_val, computed_value)
            accumulator += continuous_value*(1 - (Monomial.as_var(var) - value) ** 2)
        else:
            accumulator += get_continuous_per_formula_helper(base_var_val, evaluated_poly)* \
                           (1 - (Monomial.as_var(var) - value) ** 2)

    return accumulator


####################################################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Converter from MAX/MIN/NOT formulae to either low-degree polynomials over F_3 or a C-language simulator')
    parser.add_argument('-i', '--inputfile',
                        help='input filename containing MAX/MIN/NOT formulae. required. ',
                        type=str)
    parser.add_argument('-o', '--outputfile',
                        help='output filename for the polynomial formulae. if not provided, stdout is used',
                        type=str)
    parser.add_argument('-n',
                        action='store_true',
                        help='use descriptive names for variables')
    parser.add_argument('-no-polys',
                        action='store_true',
                        help='do not output polynomials, intended for use when output is by simulator')
    parser.add_argument('-sim',
                        action='store_true',
                        help='output C-language simulator program instead of formulae')
    parser.add_argument('--count',
                        type=int,
                        help='number of random points tried by the simulator, default 1,000,000. Ignored if the -sim flag is not used')
    parser.add_argument('-c', '--continuous',
                        action='store_true',
                        help='generate polynomials for continuous system, applied before the self-power operation')
    parser.add_argument('-comit', '--continuous-omit', nargs='+',
                        help='list of variables to _not_ apply continuity operation to')
    parser.add_argument('-power', '--self-power', type=int,
                        help='gets polynomials for a power of the system. i.e. self-composition, power-1 times ('
                             'default: 1)')
    args = parser.parse_args()

    in_formulae = ''
    if args.inputfile:
        # read input formulae from file, first checking that the file is there, etc.
        try:
            if not os.path.isfile(args.inputfile):
                print('Input file does not exist')
            elif not os.access(args.inputfile, os.R_OK):
                print('Input file is not readable')
            with open(args.inputfile, 'r') as in_file:
                in_formulae = ''.join(in_file.readlines())
        except IOError:
            print('Error reading file')
    else:
        print("Input file required\n")
        parser.print_help()

    translate_symbol_names_to_xs = not args.n

    # parse the file to (target, formula) pairs
    symbol_table, out_formulae = parse(in_formulae, translate_symbol_names_to_xs)

    # impose continuity, if desired
    if not args.continuous and args.continuous_omit:
        print("Asked to omit continuity for system which was not continuous. This is probably an error; exiting.")
        exit(-1)
    if args.continuous is not None and args.continuous:
        out_formulae = get_continuous_formulae(out_formulae, args.continuous_omit)

    # do composition if requested
    if args.self_power is not None:
        if args.self_power < 0:
            raise Exception("Cannot take negative power composition of system!")
        else:
            out_formulae = self_compose(out_formulae, args.self_power)

    if args.sim:
        count = 1_000_000
        if args.count is not None:
            count = args.count
        output_as_program(out_formulae, not args.no_polys, args.outputfile, symbol_table, count)
    else:
        output_as_formulae(out_formulae, translate_symbol_names_to_xs, not args.no_polys, args.outputfile)


def output_as_formulae(out_formulae, translate_symbol_names_to_xs, as_polynomials, output_file):
    # convert (target, formula) pairs to strings
    out_str_formulae = formulae_to_str_formulae(out_formulae, translate_symbol_names_to_xs, as_polynomials)

    if output_file:
        try:
            with open(output_file, 'w') as file_out:
                for formula in out_str_formulae:
                    file_out.write(formula)
                    file_out.write('\n')
        except IOError:
            print('Error reading file')
    else:
        for formula in out_str_formulae:
            print(formula)


def output_as_program(out_formulae, use_polys, output_file, symbol_table, num_runs):
    # get symbols, in order
    symbol_indices = sorted([symbol for symbol in symbol_table if type(symbol) == int])
    symbols = tuple((symbol_table[index] for index in symbol_indices))

    # parameter lists
    typed_param_list = "int " + ", int ".join(symbols)
    param_list = ", ".join(symbols)

    # create the functions which compute the transitions
    output_functions = []
    function_names = []

    for target, formula in out_formulae:
        if use_polys:
            string_c_formula = formula.as_polynomial().as_c_expression()
        else:
            string_c_formula = formula.as_c_expression()

        function_name = target + "_next"
        function_names.append(function_name)

        function_template = """
int {function_name}({typed_param_list})
{{
  return ( {c_formula} ) % 3;
}}"""
        function = function_template.format(function_name=function_name,
                                            typed_param_list=typed_param_list,
                                            c_formula=string_c_formula)

        output_functions.append(function)

    # random state initializer
    variable_initilization = "\n".join(
        ["    int {name}_temp, {name} = rand() % 3;".format(name=symbol)
         for symbol in symbols])

    # run update, saving to temp variables
    def update_to_temp(indent=6):
        update_template = (indent*" ") + "{symbol}_temp = {func}({params});"
        return "\n".join([update_template.format(symbol=symbol,
                                                 func=fn_name,
                                                 params=param_list)
                          for symbol, fn_name in zip(symbols, function_names)])

    # copy the temp variables over
    def copy_temp_vars(indent=6):
        copy_template = (indent*" ") + "{symbol} = {symbol}_temp;"
        return "\n".join([copy_template.format(symbol=symbol)
                          for symbol in symbols])

    # stash current "initial" state for a cycle
    def state_stash(indent=4):
        stash_template = (indent*" ") + "int {symbol}_init = {symbol};"
        return "\n".join([stash_template.format(symbol=symbol) for symbol in symbols])

    # check against "initial" state
    def neq_check(indent=4):
        neq_check_template = "{symbol} == {symbol}_init"
        return "!(" + " && ".join([neq_check_template.format(symbol=symbol) for symbol in symbols]) + ")"

    # print it out
    def print_state(indent=12):
        print_template = (indent*" ") + 'printf("\\"{symbol}\\":%u, ", {symbol});'
        print_template_last = (indent*" ") + 'printf("\\"{symbol}\\":%u ", {symbol});'
        header = (indent*" ") + 'printf("{ ");\n'
        footer = (indent*" ") + 'printf("}");\n'
        return \
          header + \
          "\n".join([print_template.format(symbol=symbol) if num != len(symbols)-1 else print_template_last.format(symbol=symbol)
                     for num, symbol in enumerate(symbols)]) + '\n' + \
          footer

    # hash functions
    hash_one = "\n".join(
        "  accumulator = 3*accumulator + {symbol};".format(symbol=symbol)
        for symbol in symbols)
    hash_two = "\n".join(
        "  accumulator = 5*accumulator + {symbol};".format(symbol=symbol)
        for symbol in symbols)
    hash_three = "\n".join(
        "  accumulator = 7*accumulator + {symbol};".format(symbol=symbol)
        for symbol in symbols)

    # load the "big" template
    with open('model-template.c', 'r') as file:
        template = file.read()

    # and use it
    program_text = template.format(param_list=param_list,
                                   typed_param_list=typed_param_list,
                                   accumulate_hash_one=hash_one,
                                   accumulate_hash_two=hash_two,
                                   accumulate_hash_three=hash_three,
                                   update_functions="\n".join(output_functions),
                                   compute_next6=update_to_temp(indent=6),
                                   compute_next8=update_to_temp(indent=8),
                                   compute_next12=update_to_temp(indent=12),
                                   copy6=copy_temp_vars(indent=6),
                                   copy8=copy_temp_vars(indent=8),
                                   copy12=copy_temp_vars(indent=12),
                                   initialize_variables=variable_initilization,
                                   print_state=print_state(),
                                   variable_stash=state_stash(indent=4),
                                   neq_check=neq_check(),
                                   num_runs=num_runs)

    if output_file:
        try:
            with open(output_file, 'w') as out_file:
                out_file.write(program_text)
        except IOError:
            print('Error writing to file')
    else:
        print(program_text)


if __name__ == '__main__':
    # execute only if run as a script
    main()
