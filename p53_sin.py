# 用"解迭代法"求解sin(x) = 1的解
import numpy
import math


def solve_sin(a, lr, epoches):
    y = lambda x: math.sin(x)
    dy_dx = lambda x: math.cos(x)
    dy = lambda x: a - y(x)
    dx = lambda x, lr: lr * dy(x) * dy_dx(x)

    x = 1.0
    for _ in range(epoches):
        x += dx(x, lr)
    return x


if __name__ == '__main__':
    print('solve_sin(1) =', solve_sin(1, lr=0.01, epoches=40000))