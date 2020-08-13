import tensorflow as tf


_name_id = 1


def unet(x, convs=5, base_filters=64, output_filters=3, name=None):
    if name is None:
        global _name_id
        name = 'unet_%d' % _name_id
        _name_id += 1
    with tf.variable_scope(name):
        semantics, stack = encode(x, convs, base_filters)
        return decode(semantics, stack, output_filters)


def encode(x, convs, filters):
    # x: [-1, h, w, 3]

    x = tf.layers.conv2d(x, filters, 3, 1, 'same', name='conv1', activation=tf.nn.relu)

    stack = []
    for i in range(convs):
        filters *= 2
        x = tf.layers.conv2d(x, filters, 3, 2, 'same', name='conv2_%d' % i, activation=tf.nn.relu)
        stack.append(x)
    return x, stack


def decode(semantics, stack, output_filters):
    stack = reversed(stack)
    filters = semantics.shape[-1].value
    y = semantics
    for i, x in enumerate(stack):
        y += x
        filters //= 2
        y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', name='deconv1_%d' % i, activation=tf.nn.relu)

    y = tf.layers.conv2d_transpose(y, output_filters, 3, 1, 'same', name='deconv2')
    return y


if __name__ == '__main__':
    # Semantics Segment
    N=20
    x = tf.random_normal([20, 128, 128, 3])
    y = unet(x, output_filters=N+1)   # [-1, 128, 128, N+1]
    label = tf.placeholder(tf.int64, [None, 128, 128])
    label = tf.one_hot(label, N+1)  # [-1, 128, 128, N+1]
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=y, dim=3)  # [-1, 128, 128]
    loss = tf.reduce_mean(loss)
    print(y.shape)
    print(loss)
