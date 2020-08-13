import tensorflow as tf

RESNET18 = 'ResNet18'
RESNET34 = 'ResNet34'
RESNET50 = 'ResNet50'
RESNET101 = 'ResNet101'
RESNET152 = 'ResNet152'

SETTING = {
    RESNET18: {'bottleneck': False, 'repeats': [2, 2, 2, 2]},
    RESNET34: {'bottleneck': False, 'repeats': [3, 4, 6, 3]},
    RESNET50: {'bottleneck': True, 'repeats': [3, 4, 6, 3]},
    RESNET101: {'bottleneck': True, 'repeats': [3, 4, 23, 3]},
    RESNET152: {'bottleneck': True, 'repeats': [3, 8, 36, 3]},
}


_name_id = 1

class TransposeResNet:
    def __init__(self, name):
        self.bottleneck = SETTING[name]['bottleneck']
        self.repeats = SETTING[name]['repeats']

    def __call__(self, x, size: int, training, name=None):
        # x: [-1, -1]
        height, width = _check(size)
        if name is None:
            global _name_id
            name = 'transpose_resnet_%d' % _name_id
            _name_id += 1
        with tf.variable_scope(name):
            filters = 2048 if self.bottleneck else 512
            x = tf.layers.dense(x, filters, name='fc', activation=tf.nn.relu)
            x = tf.reshape(x, [-1, 1, 1, filters])
            x = tf.layers.conv2d_transpose(x, filters, (height//32, width//32), 1, name='deconv1', activation=tf.nn.relu)
            x = self._repeat(x, training)  # x: [-1, 56, 56, 64]
            x = tf.layers.conv2d_transpose(x, 64, 3, 2, 'same', name='deconv2', activation=tf.nn.relu) # [-1, 112, 112, 64]
            x = tf.layers.conv2d_transpose(x, 3, (height//32, width//32), 2, 'same', name='deconv3') # [-1, 224, 224, 3]
            return x

    def _repeat(self, x, training):
        filters = x.shape[-1].value  # 2048
        for num_i, num in zip(range(len(self.repeats)-1, -1, -1), reversed(self.repeats)):
            for i in range(num-1, -1, -1):
                x = self._transpose_residual(x, num_i, i, filters, training)
            filters //= 2
        return x

    def _transpose_residual(self, x, num_i, i, filters, training):
        strides = 2 if num_i > 0 and i == 0 else 1
        if self.bottleneck:
            left = _my_deconv(x, filters, 1, 1, 'same', name='res_%d_%d_left_mydeconv1' % (num_i, i), training=training)
            filters //= 4
            left = _my_deconv(left, filters, 3, 1, 'same', name='res_%d_%d_left_mydeconv2' % (num_i, i), training=training)
            left = _my_deconv(left, filters, 1, strides, 'same', name='res_%d_%d_left_mydeconv3' % (num_i, i), training=training, active=False)
        else:
            left = _my_deconv(x, filters, 3, 1, 'same', name='res_%d_%d_left_mydeconv1' % (num_i, i), training=training)
            left = _my_deconv(left, filters, 3, strides, 'same', name='res_%d_%d_left_mydeconv2' % (num_i, i), training=training)
        if filters != x.shape[-1].value or strides > 1:
            right = _my_deconv(x, filters, 1, strides, 'same', name='res_%d_%d_right_myconv' % (num_i, i), training=training, active=False)
        else:
            right = x
        return tf.nn.relu(left + right)


def _my_deconv(x, filters, kernal_size, strides, padding, name, training, active: bool = True):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, filters, kernal_size, strides, padding, name='deconv')
        x = tf.layers.batch_normalization(x, [1, 2, 3], epsilon=1e-6, training=training, name='bn')
        if active:
            x = tf.nn.relu(x)
    return x


def _check(size):
    if type(size) == int:
        size = (size, size)
    height, width = size
    assert height % 32 == 0
    assert width % 32 == 0

    return height, width


if __name__ == '__main__':
    net = TransposeResNet(RESNET18)
    a = net(tf.random_normal([20, 123]), 224, True)
    print(a.shape)

    net = TransposeResNet(RESNET50)
    a = net(tf.random_normal([20, 123]), 224, True)
    print(a.shape)
