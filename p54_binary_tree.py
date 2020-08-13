def count_binary_tree(n, buffer={}):
    if n < 2:
        return 1

    if n in buffer:
        return buffer[n]

    n -= 1
    result = 0
    for left in range(n+1):
        result += count_binary_tree(left, buffer) * count_binary_tree(n - left, buffer)

    buffer[n+1] = result
    return result


if __name__ == '__main__':
    for i in range(1, 50+1):
        print('%d:\t%d' % (i, count_binary_tree(i)))
