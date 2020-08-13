
def sqrt(a):
    y = lambda x: x*x
    dy_dx = lambda x: 2*x
    dx = lambda x, lr:  lr * (a - y(x)) * dy_dx(x)

    x = 1
    lr = 0.001
    for _ in range(2000):
        x += dx(x, lr)
    return x


if __name__ == '__main__':
    for n in range(1, 10+1):
        print('sqrt(%s) = %f' % (n, sqrt(n)))