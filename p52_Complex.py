class Complex:
    def __init__(self, real, virtual):
        self.real = real
        self.virtual = virtual

    def __add__(self, other):
        return Complex(self.real + other.real, self.virtual + other.virtual)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.virtual - other.virtual)

    def __mul__(self, other):
        return Complex(self.real * other.real - self.virtual * other.virtual,
                       self.real * other.virtual + other.real * self.virtual)

    def __truediv__(self, other):
        a = self.real
        b = self.virtual
        c = other.real
        d = other.virtual

        devide = c ** 2 + d ** 2
        return Complex((a * c + b * d)/devide, (b * c - a * d)/devide)

    def __repr__(self):
        return "%s + %si" % (self.real, self.virtual)


if __name__ == '__main__':
    c1 = Complex(3, 4)
    c2 = Complex(2, -1)

    print('c1 + c2', c1 + c2)
    print('c1 - c2', c1 - c2)
    print('c1 * c2', c1 * c2)
    print('c1 / c2', c1 / c2)
