from enum import Enum
from itertools import product


class Operation(Enum):
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    TIMES = 'TIMES'
    EXP = 'EXP'
    MAX = 'MAX'
    MIN = 'MIN'
    CONT = 'CONT'
    NOT = 'NOT'


####################################################################################################
# monomial and sparse polynomial classes. These should be faster than the sympy versions due to
# their reduced scope.
####################################################################################################

class Expression(object):

    def __add__(self, other):
        return BinaryOperation('PLUS', self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return BinaryOperation('MINUS', self, other)

    def __mul__(self, other):
        return BinaryOperation('TIMES', self, other)

    __rmul__ = __mul__

    def __divmod__(self, other):
        raise NotImplementedError("division, modulus not implemented")

    def __neg__(self):
        raise NotImplementedError("neg not implemented")

    def __truediv__(self, other):
        raise NotImplementedError("truediv not implemented")

    def __floordiv__(self, other):
        raise NotImplementedError("floordiv not implemented")

    def __pow__(self, power, modulo=None):
        return BinaryOperation('EXP', self, power)

    def eval(self, variable_dict):
        """evaluates the expression. variable_dict is expected to be a dict containing str:Expression or
           Monomial:Expression pairs. The latter are constrained to be of single-variable type.

        """
        raise Exception("eval() unimplemented in " + str(type(self)))

    def is_constant(self):
        raise Exception("is_constant() unimplemented in " + str(type(self)))

    def as_c_expression(self):
        raise Exception("as_c_expression() unimplemented in " + str(type(self)))

    def as_polynomial(self):
        raise Exception("as_polynomial() unimplemented in " + str(type(self)))

    def get_var_set(self):
        """ returns a set containing all variable which occur in this expression """
        raise Exception("get_var_set() unimplemented in " + str(type(self)))


####################################################################################################
# actions on expressions, suitable for conversion to polynomial form. Not best for simulator.
    
def mod_3(n):
    return n % 3

def not3(n):
    value = 2 + 2*n
    if type(value) == int or value.is_constant():
        return mod_3(int(value))
    else:
        return value

def max3(a, b):
    value = a + b + 2*a*b + (a ** 2)*b + a*(b ** 2) + (a ** 2)*(b ** 2)
    if type(value) == int or value.is_constant():
        return mod_3(int(value))
    else:
        return value

def min3(a, b):
    value = a*b + 2*(a ** 2)*b + 2*a*(b ** 2) + 2*(a ** 2)*(b ** 2)
    if type(value) == int or value.is_constant():
        return mod_3(int(value))
    else:
        return value


def h(x, fx):
    """helper function as in the PLoS article, doi:10.1371/journal.pcbi.1005352.t003 pg 16/24"""
    if fx > x:
        return x + 1
    elif fx < x:
        return x - 1
    else:
        return x
    
def cont3(ctrl_var, formula):
    """convert the expression expr into one which is 'continuous' in the
    control variable ctrl"""
    # handle case where the control variable is a constant
    if type(ctrl_var) == int or ctrl_var.is_constant():
        ctrl_var = int(ctrl_var) % 3
        if type(formula) == int or formula.is_constant():
            formula = int(formula) % 3
            return h(ctrl_var, formula)
        else:
            return cont3_helper(ctrl_var, formula)

    # otherwise, the control variable must be just a variable
    assert type(ctrl_var) == str or \
      ( type(ctrl_var) == Monomial and len(ctrl_var.get_var_set()) == 1 )

    # go through the whole buisness for the target variable first
    accumulator = Mod3Poly.zero()
    for base_value in range(3):
        if not( type(formula) == int or formula.is_constant() ):
            evaluated_poly = formula.eval({ctrl_var: base_value})
        if type(evaluated_poly) == int or evaluated_poly.is_constant():
            computed_value = int(evaluated_poly)
            continuous_value = h(base_value, computed_value)
            accumulator += continuous_value*(1 - (Monomial.as_var(ctrl_var) - base_value) ** 2)
        else:
            accumulator += cont3_helper(base_value, evaluated_poly) * (1 - (Monomial.as_var(ctrl_var) - base_value) ** 2)
    return accumulator
    
def cont3_helper(base_var_val, formula):
    """helper function for cont3, gets continuous version of the formula for variable var"""
    # find some unevalued variable
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
            accumulator += cont3_helper(base_var_val, evaluated_poly) * (1 - (Monomial.as_var(var) - value) ** 2)

    return accumulator
    

####################################################################################################

class Function(Expression):

    def __init__(self, function_name, expression_list):
        self._function_name = function_name
        self._expression_list = expression_list

    def eval(self, variable_dict):
        # evaluate function parameters
        evaluated_expressions = [expr if type(expr) == int
                                 else expr.eval(variable_dict)
                                 for expr in self._expression_list]
        # simplify constants to ints, if possible
        evaluated_expressions = [int(expr) if type(expr) == int or expr.is_constant()
                                 else expr
                                 for expr in evaluated_expressions]

        if self._function_name == 'MAX':
            assert len(evaluated_expressions) == 2, "wrong number of arguments for MAX"
            exprOne, exprTwo = evaluated_expressions
            # if it can be computed directly, do it. otherwise, return in function form
            if type(exprOne) == int and type(exprTwo) == int:
                exprOne = mod_3(exprOne)
                exprTwo = mod_3(exprTwo)
                return max(exprOne, exprTwo)
            else:
                return Function('MAX', [exprOne, exprTwo])
        elif self._function_name == 'MIN':
            assert len(evaluated_expressions) == 2, "wrong number of arguments for MIN"
            exprOne, exprTwo = evaluated_expressions
            # if it can be computed directly, do it. otherwise, return in function form
            if type(exprOne) == int and type(exprTwo) == int:
                exprOne = mod_3(exprOne)
                exprTwo = mod_3(exprTwo)
                return min(exprOne, exprTwo)
            else:
                return Function('MIN', [exprOne, exprTwo])
        elif self._function_name == 'CONT':
            assert len(evaluated_expressions) == 2, "wrong number of arguments for CONT"
            ctrl_var, expr = evaluated_expressions
            if type(ctrl_var) == int and type(expr) == int:
                ctrl_var = mod_3(ctrl_var)
                expr = mod_3(expr)
                return XXX
            # cannot be computed directly, so return in function form
            return Function('CONT', [ctrl_var, expr])
        elif self._function_name == 'NOT':
            assert len(evaluated_expressions) == 1, "wrong number of arguments for NOT"
            expr = evaluated_expressions[0]
            # if it can be computed directly, do it. otherwise, return in function form
            if type(expr) == int:
                return not3(expr)
            else:
                return Function('NOT', [expr])
        else:
            raise Exception("cannot evaluate unknown function " + self._function_name)

    def is_constant(self):
        return all(type(expr) == int or expr.is_constant()
                   for expr in self._expression_list)

    def __str__(self):
        return self._function_name + "(" + ",".join([str(exp) for exp in self._expression_list]) + ")"

    __repr__ = __str__

    def as_c_expression(self):
        c_exprs = [str(expr) if type(expr) == int else expr.as_c_expression() for expr in self._expression_list]

        if self._function_name == 'MAX':
            func_name = 'mod3max'
        elif self._function_name == 'MIN':
            func_name = 'mod3min'
        elif self._function_name == 'CONT':
            func_name = 'mod3continuity'
        elif self._function_name == 'NOT':
            func_name = 'mod3not'
        else:
            raise Exception("Unknown binary relation: " + self._function_name)

        return func_name + '(' + ",".join(c_exprs) + ')'

    def as_polynomial(self):
        expressions_as_polynomials = [mod_3(expr) if type(expr) == int
                                      else expr.as_polynomial()
                                      for expr in self._expression_list]

        if self._function_name == 'MAX':
            assert len(expressions_as_polynomials) == 2, "wrong number of arguments for MAX"
            return max3(expressions_as_polynomials[0], expressions_as_polynomials[1])

        elif self._function_name == 'MIN':
            assert len(expressions_as_polynomials) == 2, "wrong number of arguments for MIN"
            return min3(expressions_as_polynomials[0], expressions_as_polynomials[1])

        elif self._function_name == 'CONT':
            assert len(expressions_as_polynomials) == 2, "wrong number of arguments for CONT"
            return cont3(expressions_as_polynomials[0], expressions_as_polynomials[1])

        elif self._function_name == 'NOT':
            assert len(expressions_as_polynomials) == 1, "wrong number of arguments for NOT"
            return not3(expressions_as_polynomials[0])

        else:
            raise Exception("cannot evaluate unknown function " + self._function_name + " as a polynomial")

    def get_var_set(self):
        var_set = set()
        for expr in self._expression_list:
            if type(expr) != int:
                var_set = var_set.union(expr.get_var_set())
        return var_set


class BinaryOperation(Expression):

    def __init__(self, relation_name, left_expression, right_expression):
        self._relation_name = relation_name
        self._left_expression = left_expression
        self._right_expression = right_expression

    def is_constant(self):
        return (type(self._left_expression) == int or self._left_expression.is_constant()) and \
               (type(self._right_expression) == int or self._right_expression.is_constant())

    def eval(self, variable_dict):
        # evaluate parameters, making them ints if possible
        evaled_left_expr = self._left_expression if type(self._left_expression) == int \
            else self._left_expression.eval(variable_dict)
        evaled_left_expr = int(evaled_left_expr) \
            if type(evaled_left_expr) == int or evaled_left_expr.is_constant() \
            else evaled_left_expr

        evaled_right_expr = self._right_expression if type(self._right_expression) == int \
            else self._right_expression.eval(variable_dict)
        evaled_right_expr = int(evaled_right_expr) \
            if type(evaled_right_expr) == int or evaled_right_expr.is_constant() \
            else evaled_right_expr

        if self._relation_name == 'PLUS':
            return evaled_left_expr + evaled_right_expr
        elif self._relation_name == 'MINUS':
            return evaled_left_expr - evaled_right_expr
        elif self._relation_name == 'TIMES':
            return evaled_left_expr*evaled_right_expr
        elif self._relation_name == 'EXP':
            return evaled_left_expr ** evaled_right_expr
        else:
            raise Exception("cannot evaluate unknown binary op: " + self._relation_name)

    def __str__(self):
        short_relation_name = "?"
        if self._relation_name == 'PLUS':
            short_relation_name = '+'
        elif self._relation_name == 'MINUS':
            short_relation_name = '-'
        elif self._relation_name == 'TIMES':
            short_relation_name = '*'
        elif self._relation_name == 'EXP':
            short_relation_name = '^'

        left_side = str(self._left_expression)
        if type(self._left_expression) == BinaryOperation:
            left_side = "(" + left_side + ")"

        right_side = str(self._right_expression)
        if type(self._right_expression) == BinaryOperation:
            right_side = "(" + right_side + ")"

        return left_side + short_relation_name + right_side

    __repr__ = __str__

    def as_c_expression(self):
        if type(self._left_expression) == int:
            left_c_expr = str(self._left_expression)
        else:
            left_c_expr = self._left_expression.as_c_expression()

        if type(self._right_expression) == int:
            right_c_expr = str(self._right_expression)
        else:
            right_c_expr = self._right_expression.as_c_expression()

        if self._relation_name == 'PLUS':
            return '(' + left_c_expr + ')+(' + right_c_expr + ')'

        elif self._relation_name == 'MINUS':
            return '(' + left_c_expr + ')-(' + right_c_expr + ')'

        elif self._relation_name == 'TIMES':
            return '(' + left_c_expr + ')*(' + right_c_expr + ')'

        elif self._relation_name == 'EXP':
            return 'mod3pow(' + left_c_expr + ',' + right_c_expr + ')'

        else:
            raise Exception("Unknown binary relation: " + self._relation_name)

    def as_polynomial(self):
        if type(self._left_expression) == int:
            left_poly = self._left_expression
        else:
            left_poly = self._left_expression.as_polynomial()

        if type(self._right_expression) == int:
            right_poly = self._right_expression
        else:
            right_poly = self._right_expression.as_polynomial()

        if self._relation_name == 'PLUS':
            return left_poly + right_poly

        elif self._relation_name == 'MINUS':
            return left_poly - right_poly

        elif self._relation_name == 'TIMES':
            return left_poly*right_poly

        elif self._relation_name == 'EXP':
            # simplify the exponent = 0, 1 cases
            if type(right_poly) == int:
                if right_poly == 0:
                    return 1
                elif right_poly == 1:
                    return left_poly
                else:
                    return left_poly ** right_poly
            else:
                return left_poly ** right_poly
        else:
            raise Exception("Unknown binary relation: " + self._relation_name)

    def get_var_set(self):
        var_set = set()
        if type(self._left_expression) != int:
            var_set = var_set.union(self._left_expression.get_var_set())
        if type(self._right_expression) != int:
            var_set = var_set.union(self._right_expression.get_var_set())
        return var_set


class UnaryRelation(Expression):

    def __init__(self, relation_name, expr):
        self._relation_name = relation_name
        self._expr = expr

    def __str__(self) -> str:
        shortRelName = str(self._relation_name)
        if self._relation_name == 'MINUS':
            shortRelName = '-'
        return shortRelName + \
               ("(" + str(self._expr) + ")" if type(self._expr) == BinaryOperation else str(self._expr))

    __repr__ = __str__

    def as_c_expression(self):
        if type(self._expr) == int:
            c_exp = str(mod_3(self._expr))
        else:
            c_exp = self._expr.as_c_expression()

        if self._relation_name == 'MINUS':
            return '-(' + c_exp + ')'
        else:
            raise Exception("Unknown binary relation: " + self._relation_name)

    def as_polynomial(self):
        if type(self._expr) == int or self._expr.is_constant():
            poly = mod_3(int(self._expr))
        else:
            poly = self._expr.as_polynomial()

        if self._relation_name == 'MINUS':
            return - poly
        else:
            raise Exception("Unknown unary relation: " + self._relation_name)

    def get_var_set(self):
        if type(self._expr) == int:
            return set()
        else:
            return self._expr.get_var_set()


####################################################################################################

class Monomial(Expression):
    """A class to encapsulate monomials reduced by x^3-x==0 for all variables x"""

    def __init__(self, power_dict: dict):
        # copy over only those terms which actually appear
        self._power_dict = {str(var): power_dict[var] for var in power_dict if power_dict[var] != 0}
        for var in self._power_dict.keys():
            #while self._power_dict[var] < 0:
            #    self._power_dict[var] += 2     <--- replace with below
            assert self._power_dict[var] > 0 # b/c x^-1 isn't exactly x (i.e. when x=0)
            #while self._power_dict[var] >= 3:
            #    self._power_dict[var] -= 2     <--- replace with below
            self._power_dict[var] = 1 + ((-1+self._power_dict[var]) % 2)

    def as_polynomial(self):
        return self

    def is_constant(self):
        return len(self._power_dict) == 0

    def eval(self, variable_dict):
        """evaluates the monomial. variable_dict is expected to be a dict containing str:Expression or
           Monomial:Expression pairs. The latter are constrained to be of single-variable type.

        """
        if type(variable_dict) != dict:
            raise Exception("eval is not defined on this input")

        # sanitize inputs
        sanitized_variable_dict = dict()
        for variable, quantity in variable_dict.items():
            if type(variable) == str:
                sanitized_variable_dict.update({variable: variable_dict[variable]})
            elif type(variable) == Monomial:
                if len(variable._power_dict) != 1:
                    raise Exception("We do not know how to evaluate monomials of zero or several variables")
                else:
                    variable_as_str = list(variable._power_dict.keys())[0]
                    sanitized_variable_dict.update({variable_as_str: variable_dict[variable]})
        variable_dict = sanitized_variable_dict

        accumulator = Mod3Poly.one()
        for variable, quantity in self._power_dict.items():
            if variable in variable_dict.keys():
                accumulator *= variable_dict[variable] ** self._power_dict[variable]
            else:
                accumulator *= Monomial.as_var(variable) ** self._power_dict[variable]
        return accumulator

    def get_var_set(self):
        """ returns a set containing all variable which occur in this monomial """
        return {var for var in self._power_dict if self._power_dict[var] != 0}

    @staticmethod
    def unit():
        """produces the unit, 1, as a monomial"""
        return Monomial(dict())

    @staticmethod
    def as_var(var_name: str):
        return Monomial({var_name: 1})

    def __mul__(self, other) -> Expression:
        if isinstance(other, Monomial):
            result_power_dict = self._power_dict.copy()
            for key in other._power_dict.keys():
                if key in result_power_dict.keys():
                    result_power_dict[key] += other._power_dict[key]
                    while result_power_dict[key] >= 3:
                        result_power_dict[key] -= 2
                else:
                    result_power_dict[key] = other._power_dict[key]
            return Monomial(result_power_dict)
        elif isinstance(other, Mod3Poly) or isinstance(other, int):
            return self.as_poly()*other
        else:
            return BinaryOperation('TIMES', self, other)
            # raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(self.__class__, type(other)))

    __rmul__ = __mul__

    def __neg__(self):
        return (-1)*self

    def __pow__(self, power, **kwargs):
        if type(power) == Mod3Poly and power.is_constant():
            power = power[Monomial.unit()]
        assert type(power) == int
        if power == 0:
            return Monomial.unit()
        elif power == 1:
            return self
        elif power == 2:
            return self*self
        # Now handle higher powers; probably not going to happen too much for this application

        # (int) half power root
        int_root = self ** (power//2)
        if power%2 == 0:
            return int_root*int_root
        else:
            return int_root*int_root*self

    def as_poly(self):
        """converts this monomial to a polynomial with only one term"""
        return Mod3Poly({self: 1})

    def __add__(self, other):
        if type(other) == Mod3Poly:
            return other + self.as_poly()
        elif type(other) == Monomial:
            return self.as_poly() + other.as_poly()
        elif type(other) == int:
            return self.as_poly() + other
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + ((-1)*other)

    def __rsub__(self, other):
        return ((-1)*self) + other

    def __eq__(self, other):
        if type(other) == str:
            other = Monomial.as_var(other)
        if type(other) == Monomial:
            return self._power_dict == other._power_dict
        elif type(other) == Mod3Poly:
            if len(other.coeff_dict) == 1:
                monomial, coeff = list(other.coeff_dict)[0]
                return coeff == 1 and monomial == self
            else:
                return False
        elif type(other) == int and self == Monomial.unit():
            return other == 1
        else:
            return False

    def __ne__(self, other):
        if type(other) == str:
            other = Monomial.as_var(other)
        return not (self == other)

    def __lt__(self, other):
        self_vars = set(self._power_dict.keys())
        if type(other) == str:
            other = Monomial.as_var(other)
        other_vars = set(other._power_dict.keys())
        # if we have a var that they don't we cannot be "smaller"
        if len(self_vars - other_vars) > 0:
            return False
        # check that we do not exceed and are smaller at least once
        at_least_once_less = False
        for var in self_vars:
            if self._power_dict[var] > other._power_dict[var]:
                return False
            elif self._power_dict[var] < other._power_dict[var]:
                at_least_once_less = True
        return at_least_once_less or len(other_vars - self_vars) > 0

    def __le__(self, other):
        self_vars = set(self._power_dict.keys())
        if type(other) == str:
            other = Monomial.as_var(other)
        other_vars = set(other._power_dict.keys())
        # if we have a var that they don't we cannot be "smaller"
        if len(self_vars - other_vars) > 0:
            return False
        # check that we do not exceed 
        for var in self_vars:
            if self._power_dict[var] > other._power_dict[var]:
                return False
        return True

    def __gt__(self, other):
        self_vars = set(self._power_dict.keys())
        if type(other) == str:
            other = Monomial.as_var(other)
        other_vars = set(other._power_dict.keys())
        # if they have a var that they don't we cannot be "greater"
        if len(other_vars - self_vars) > 0:
            return False
        # check that we are not smaller and are greater at least once
        at_least_once_greater = False
        for var in other_vars:
            if self._power_dict[var] < other._power_dict[var]:
                return False
            elif self._power_dict[var] > other._power_dict[var]:
                at_least_once_greater = True
        return at_least_once_greater or len(self_vars - other_vars) > 0

    def __ge__(self, other):
        self_vars = set(self._power_dict.keys())
        if type(other) == str:
            other = Monomial.as_var(other)
        other_vars = set(other._power_dict.keys())
        # if they have a var that they don't we cannot be "greater"
        if len(other_vars - self_vars) > 0:
            return False
        # check that we are not smaller
        for var in other_vars:
            if self._power_dict[var] < other._power_dict[var]:
                return False
        return True

    def __hash__(self):
        return sum(hash(k) for k in self._power_dict.keys()) + \
               sum(hash(v) for v in self._power_dict.values())

    def __str__(self):
        if self._power_dict == {}:
            return "1"
        else:
            variables = sorted(self._power_dict.keys())
            return "*".join([str(var) + "^" + str(self._power_dict[var])
                             if self._power_dict[var] > 1 else str(var) for var in variables])

    __repr__ = __str__

    def as_c_expression(self):
        if self._power_dict == {}:
            return "1"
        else:
            variables = sorted(self._power_dict.keys())
            return "*".join(["mod3pow(" + str(var) + "," + str(self._power_dict[var]) + ")"
                             if self._power_dict[var] > 1 else str(var) for var in variables
                             if self._power_dict[var] != 0])


####################################################################################################

class Mod3Poly(Expression):
    """a sparse polynomial class"""

    def __init__(self, coeffs):
        if type(coeffs) == dict:
            self.coeff_dict = {monomial: coeffs[monomial] for monomial in coeffs if coeffs[monomial] != 0}
        elif type(coeffs) == int:
            self.coeff_dict = {Monomial.unit(): (coeffs%3)}
        else:
            raise TypeError("unsupported initialization type for '{}': '{}'".format(self.__class__, type(coeffs)))

    @staticmethod
    def zero():
        return Mod3Poly({Monomial.unit(): 0})

    @staticmethod
    def one():
        return Mod3Poly({Monomial.unit(): 1})

    def as_polynomial(self):
        return self

    def __int__(self):
        self.__clear_zero_monomials()
        if len(self.coeff_dict) > 1 or (len(self.coeff_dict) == 1 and Monomial.unit() not in self.coeff_dict):
            raise Exception("cannot cast non-constant polynomial to int")
        if Monomial.unit() in self.coeff_dict:
            return self.coeff_dict[Monomial.unit()]
        else:
            return 0

    def eval(self, variable_dict):
        """evaluates the polynomial. variable_dict is expected to be a dict containing str:Expression or
           Monomial:Expression pairs. The latter are constrained to be of single-variable type. """
        if type(variable_dict) != dict:
            raise Exception("Mod3Poly.eval is not defined on this input")

        accumulator = Mod3Poly.zero()
        for monomial, coeff in self.coeff_dict.items():
            accumulator += coeff*monomial.eval(variable_dict)
        return accumulator

    def get_var_set(self):
        """return a set containing all variables which occur in this polynomial"""
        var_set = set()
        for monomial in self.coeff_dict:
            var_set = var_set.union(monomial.get_var_set())
        return var_set

    def __clear_zero_monomials(self):
        """purge unneeded data"""
        self.coeff_dict = {monom: self.coeff_dict[monom] for monom in self.coeff_dict if self.coeff_dict[monom] != 0}
        # assure at least one entry
        if len(self.coeff_dict) == 0:
            self.coeff_dict = {Monomial.unit(): 0}

    def is_constant(self):
        # possibly unnecessary 
        self.__clear_zero_monomials()
        num_nonzero_monom = len(self.coeff_dict)
        if num_nonzero_monom > 1:
            return False
        elif num_nonzero_monom == 0:
            return True
        else:
            # only one entry
            return Monomial.unit() in self.coeff_dict

    def __getitem__(self, index):
        if index in self.coeff_dict:
            return self.coeff_dict[index]
        else:
            return 0

    def __setitem__(self, index, value):
        self.coeff_dict[index] = value

    def __add__(self, other):
        if type(other) == int:
            self_copy = Mod3Poly(self.coeff_dict)
            self_copy[Monomial.unit()] = (self_copy[Monomial.unit()] + other)%3
            return self_copy
        elif type(other) == Mod3Poly:
            self_copy = Mod3Poly(self.coeff_dict)
            for key in other.coeff_dict.keys():
                if key in self_copy.coeff_dict.keys():
                    self_copy[key] = (self_copy[key] + other[key])%3
                else:
                    self_copy[key] = other[key]
            return self_copy
        elif type(other) == Monomial:
            self_copy = Mod3Poly(self.coeff_dict)
            self_copy[other] += 1
            return self_copy
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    __radd__ = __add__

    def __sub__(self, other):
        if type(other) == int:
            self_copy = Mod3Poly(self.coeff_dict)
            self_copy[Monomial.unit()] = (self_copy[Monomial.unit()] - other)%3
            return self_copy
        elif type(other) == Mod3Poly or type(other) == Monomial:
            self_copy = Mod3Poly(self.coeff_dict)
            if type(other) == Monomial:
                other = other.as_poly()
            for key in other.coeff_dict.keys():
                if key in self_copy.coeff_dict.keys():
                    self_copy[key] = (self_copy[key] - other[key])%3
                else:
                    self_copy[key] = other[key]
            return self_copy
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    def __rsub__(self, other):
        return other + ((-1)*self)

    def __mul__(self, other):
        if type(other) == int:
            return Mod3Poly({key: (self.coeff_dict[key]*other)%3 for key in self.coeff_dict})
        elif type(other) == Monomial:
            return Mod3Poly({(other*monomial): self.coeff_dict[monomial] for monomial in self.coeff_dict})
        elif type(other) == Mod3Poly:
            accumulator = Mod3Poly.zero()
            for self_mono, other_mono in product(self.coeff_dict.keys(), other.coeff_dict.keys()):
                monomial_prod = self_mono*other_mono
                accumulator[monomial_prod] = (accumulator[monomial_prod] + self[self_mono]*other[other_mono])%3
            return accumulator
        else:
            return BinaryOperation('TIMES', self, other)

    __rmul__ = __mul__

    def __neg__(self):
        return (-1)*self

    def __pow__(self, power, **kwargs):
        if type(power) == Mod3Poly and power.is_constant():
            power = power[Monomial.unit()]
        assert type(power) == int
        if power == 0:
            return Monomial.unit().as_poly()
        elif power == 1:
            return self
        elif power == 2:
            return self*self
        # Now handle higher powers; probably not going to happen too much for this application

        # (int) half power root
        int_root = self ** (power//2)
        if power%2 == 0:
            return int_root*int_root
        else:
            return int_root*int_root*self

    def __str__(self):
        accumulator = ""
        for monomial in sorted(self.coeff_dict.keys()):
            if monomial == Monomial.unit():
                if self[monomial] != 0:
                    accumulator += str(self[monomial])
            else:
                if len(accumulator) > 0 and self[monomial] != 0:
                    accumulator += "+"
                if self[monomial] == 1:
                    accumulator += str(monomial)
                elif self[monomial] == 2:
                    accumulator += "2*"
                    accumulator += str(monomial)
        if len(accumulator) > 0:
            return accumulator
        else:
            return "0"

    __repr__ = __str__

    def as_c_expression(self):
        accumulator = ""
        for monomial in sorted(self.coeff_dict.keys()):
            if monomial == Monomial.unit():
                if self[monomial] != 0:
                    accumulator += str(self[monomial])
            else:
                if len(accumulator) > 0 and self[monomial] != 0:
                    accumulator += "+"
                if self[monomial] == 1:
                    accumulator += monomial.as_c_expression()
                elif self[monomial] == 2:
                    accumulator += "2*"
                    accumulator += monomial.as_c_expression()
        if len(accumulator) > 0:
            return accumulator
        else:
            return "0"
