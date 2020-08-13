import math


class Exp:
    def eval(self, **values):
        pass

    def simplify(self):  # 把当前表达式转成一个更简单的表达式
        return self

    def deriv(self, x):  # 求偏导
        pass

    def __add__(self, other):
        return Add(self, to_exp(other)).simplify()

    def __radd__(self, other):
        return Add(to_exp(other), self).simplify()

    def __sub__(self, other):
        return Sub(self, to_exp(other)).simplify()

    def __rsub__(self, other):
        return Sub(to_exp(other), self).simplify()

    def __mul__(self, other):
        return Mul(self, to_exp(other)).simplify()

    def __rmul__(self, other):
        return Mul(to_exp(other), self).simplify()

    def __truediv__(self, other):
        return TrueDiv(self, to_exp(other)).simplify()

    def __rtruediv__(self, other):
        return TrueDiv(to_exp(other), self).simplify()

    def __neg__(self):
        return Neg(self).simplify()

    def __pow__(self, power, modulo=None):
        return Pow(self, to_exp(power)).simplify()

    def __rpow__(self, other):
        return Pow(to_exp(other), self).simplify()


class Pow(Exp):
    def __init__(self, base, power):
        self.base = base
        self.power = power

    def eval(self, **values):
        return self.base.eval(**values) ** self.power.eval(**values)

    def simplify(self):
        if isinstance(self.power, Const):
            if self.power.value == 0:
                return Const(1)
            if self.power.value == 1:
                return self.base
            if isinstance(self.base, Const):
                return Const(self.base.value ** self.power.value)
        elif isinstance(self.base, Const) and self.base.value in (0, 1):
            return Const(self.base.value)
        return self

    def deriv(self, x):  # (u**v)' = y * v' * ln(u) + v * u**(v-1) * u'
        u, v = self.base, self.power
        return self * v.deriv(x) * log(u) + v * u ** (v - 1) * u.deriv(x)

    def __repr__(self):
        return '(%s ** %s)' % (self.base, self.power)


class Neg(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        return -self.value.eval(**values)

    def simplify(self):
        if isinstance(self.value, Const):
            return Const(-self.value.value)
        return self

    def deriv(self, x):
        return -self.value.deriv(x)

    def __repr__(self):
        return '(-%s)' % self.value


def to_exp(value):
    if isinstance(value, Exp):
        return value
    elif type(value) in (float, int):
        return Const(value)
    else:
        raise Exception('Can not convert %s into Exp' % value)


class Const(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        return self.value

    def deriv(self, x):
        return Const(0)

    def __repr__(self):
        return str(self.value)


e = Const(math.e)


def log(value, base=e):
    return Log(value, base).simplify()


class Log(Exp):
    def __init__(self, value, base):
        self.value = value
        self.base = base

    def eval(self, **values):
        return math.log(self.value.eval(**values), self.base.eval(**values))

    def simplify(self):
        if isinstance(self.value, Const):
            if self.value.value == 1:
                return Const(0)
            if isinstance(self.base, Const):
                return Const(math.log(self.value.value, self.base.value))
        return self

    def deriv(self, x):  # (log(u, v))' = (u' * ln(v)/u - v' * ln(u)/v) / (ln(v)**2)
        u, v = self.value, self.base
        result = u.deriv(x) * log(v) / u - v.deriv(x) * log(u) / v
        return result / log(v) ** 2


class Variable(Exp):
    def __init__(self, name):
        self.name = name

    def eval(self, **values):
        if self.name in values:
            return values[self.name]
        raise Exception('Variable %s is not found.' % self.name)

    def deriv(self, x):
        name = _get_name(x)
        return Const(1 if name == self.name else 0)

    def __repr__(self):
        return self.name


def _get_name(x):
    if isinstance(x, Variable):
        return x.name
    if type(x, str):
        return x
    raise Exception('%x can not be used to get derivant from an expression' % x)


class Add(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left = left
        self.right = right

    def deriv(self, x):
        return self.left.deriv(x) + self.right.deriv(x)

    def simplify(self):
        left, right = self.left, self.right
        if isinstance(left, Const):
            if left.value == 0:
                return right
            if isinstance(right, Const):
                return Const(left.value + right.value)
        elif isinstance(right, Const) and right.value == 0:
            return left
        return self

    def eval(self, **values):
        return self.left.eval(**values) + self.right.eval(**values)

    def __repr__(self):
        return '(%s + %s)' % (self.left, self.right)


class Sub(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left = left
        self.right = right

    def deriv(self, x):
        return self.left.deriv(x) - self.right.deriv(x)

    def eval(self, **values):
        return self.left.eval(**values) - self.right.eval(**values)

    def __repr__(self):
        return '(%s - %s)' % (self.left, self.right)

    def simplify(self):
        left, right = self.left, self.right
        if isinstance(left, Const):
            if left.value == 0:
                return -right
            if isinstance(right, Const):
                return Const(left.value - right.value)
        elif isinstance(right, Const) and right.value == 0:
            return left
        return self


class Mul(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left = left
        self.right = right

    def deriv(self, x):  # (uv)' = u'v + uv'
        u, v = self.left, self.right
        return u.deriv(x) * v + u * v.deriv(x)

    def simplify(self):
        left, right = self.left, self.right
        if isinstance(left, Const):
            if left.value == 0:
                return Const(0)
            elif left.value == 1:
                return right
            if isinstance(right, Const):
                return Const(left.value * right.value)
        elif isinstance(right, Const):
            if right.value == 0:
                return Const(0)
            elif right.value == 1:
                return left
        return self

    def eval(self, **values):
        return self.left.eval(**values) * self.right.eval(**values)

    def __repr__(self):
        return '(%s * %s)' % (self.left, self.right)


class TrueDiv(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left = left
        self.right = right

    def deriv(self, x):  # (u/v)' = (u'v - uv')/v**2
        u, v = self.left, self.right
        return (u.deriv(x) * v - u * v.deriv(x)) / v ** 2

    def simplify(self):
        left, right = self.left, self.right
        if isinstance(left, Const):
            if left.value == 0:
                return Const(0)
            if isinstance(right, Const):
                return Const(left.value / right.value)
        elif isinstance(right, Const):
            if right.vlaue == 0:
                raise Exception('Divided by zero!')
            elif right.value == 1:
                return left
        return self

    def eval(self, **values):
        return self.left.eval(**values) / self.right.eval(**values)

    def __repr__(self):
        return '(%s / %s)' % (self.left, self.right)


def sin(value):
    return Sin(to_exp(value)).simplify()


class Sin(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        return math.sin(self.value.eval(**values))

    def simplify(self):
        if isinstance(self.value, Const):
            return Const(math.sin(self.value.value))
        return self

    def deriv(self, x):
        return cos(self.value) * self.value.deriv(x)

    def __repr__(self):  # representation
        return 'sin(%s)' % self.value


if __name__ == '__main__':
    c1 = Const(1)
    c2 = Const(12.345)
    print(c1.eval(), c2.eval())
    print(c1, c2)

    x = Variable('x')
    y = Variable('y')
    print(x.eval(x=123.456), y.eval(x=123.456, y=789.012))

    print('c1 + c2 = ', c1 + c2)
    print('c1 + c2 * c1 = ', c1 + c2 * c1)
    print('(c1 + c2) * c2 = ', (c1 + c2) * c2)

    print('c1 + x = ', (c1 + x).eval(x=10))

    print((c1 * (x + y)).eval(x=10, y=20))

    print(c1 + 3)
    print(3 + x)  # === x.__radd__(3)
    print(3 - c1)

    print(3 - 3 - c1)

    print((3 * x + 4).deriv(x))

    a = (3 * x ** 5 + 4 * x + 12).deriv(x)
    print(a, '(x=0.5)=', a.eval(x=0.5))

    b = e ** (-3 * x ** 2 - 3 * x + 4)
    a = b.deriv(x)
    print(b)
    print(a)

    print((x ** 0.5).deriv(x))

    print(sin(math.pi / 2))

    print((x ** 2 + y ** 2).deriv(y).eval(y=0.5))
