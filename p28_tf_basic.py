import tensorflow as tf
import numpy as np

# print(np.random.normal([4, 3]) + np.random.uniform(0, 10, [2, 3]))

c = tf.constant(123, dtype=tf.float32, name='abc')
c2 = tf.constant([[1, 2, 3], [4, 5, 6]], name='def')

c3 = tf.random_normal([2, 3, 4])
c4 = tf.random_uniform([2, 3, 1], 0, 10)   # [0, 10)
c5 = tf.random_normal([3, 4])
c6 = tf.random_normal([3, 2])

v1 = tf.Variable(123, name='v1')
v2 = tf.get_variable('v2', [3, 4], tf.float32)
v3 = tf.get_variable('v3', [4, 9], tf.float32)
# 1) always use get_variable
# 2) The value of a variable can be reserved from a session's run to another one.
# 3) Please initialize a variable before use it.

print(c.shape)
print(c2.shape)

d = c + 3
d2 = c2 + 3


#    算术运算：+，-，*， /， //， %， **, >>, <<
#    关系运算：>, <, >=, <=, ==, !=

# tf.add(a, b) === a + b
tf.logical_and
tf.logical_or
tf.logical_not
tf.logical_xor

tf.cast
np.cast

tf.matmul
tf.concat
np.concatenate

tf.reshape

v4 = tf.reshape(v3, [2, 2, 3, 1, 3, 1, 1, 1, 1])
v5 = tf.reshape(v3, [2, 2, -1, 3])
# v6 = tf.reshape(v5, [7, 9, -1])
v7 = tf.expand_dims(v3, axis=-1)
print(v7.shape)
v8 = tf.expand_dims(v3, axis=1)
print(v8.shape)
print('=' * 30)

v = tf.random_normal([2,3,5,7])
v9 = tf.transpose(v)
print(v9.shape)
v10 = tf.transpose(v, [1, 0, 2, 3])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    c_v = session.run(c)
    print(c_v)
    print(session.run(c2))
    print(session.run(d))
    d2_v = session.run(d2)
    print(type(d2_v))
    print(d2_v)

    print('-' * 30)
    print(session.run([c3, c4, c3 + c4]))
    print('-' * 30)
    print(session.run([c3, c5, c3 > c5]))
    # print(session.run([c3, c6, c3 > c6]))

    print(session.run(v2))
    print(session.run(v2+3))

    print(session.run(tf.matmul(v2, v3)))
    print(session.run(tf.matmul(v2, v3)).shape)

(3, 4)


# broadcasting
# 1)  a consistant with b if a.shape == b.shape
# 2)  a consistant with b if a.shape == ()
# 3)  a.shape consistants with b.shape[:i] + [1] + b.shape[i+1:] if a consistants with b
# 4)  a.shape consistants with  b.sahpe[1:] if a consistants with b
# 5)  b consistants with a if a consistants with b

p = tf.placeholder(tf.float32, [3, 5, 7], name='p')
a = p * 3
with tf.Session() as session:
    a_v = session.run(a, {p: np.random.uniform(0, 5, [3, 5, 7])})
    print(a_v)

# loss = tf.reduce_mean((a - m)**2)
# lr = 0.01
# opt = tf.train.GradientDescentOptimizer(lr)
# opt.minimize(loss)