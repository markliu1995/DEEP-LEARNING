import p50_framework as myf
import p55_VAE_face as face
import p56_ResNet as resnet
import p58_transpose_ResNet as deresnet
import tensorflow as tf


class MyConfig(face.MyConfig):
    def get_name(self):
        return 'p59'

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_tensors(self):
        return myf.Tensors(self)


class MySubTensors(face.MySubTensors):
    def get_loss(self, x):
        # x: [-1, h, w, 3]
        loss = tf.reduce_mean(tf.abs(self.y - x))
        length = tf.shape(self.y)[0] // 2        # x.shape[-1].value
        y1 = self.y[0:length, :, :, :]
        y2 = self.y[length:, :, :, :]
        x1 = x[:length, :, :, :]
        x2 = x[length:, :, :, :]
        loss2 = tf.reduce_mean(tf.abs(y1-y2-(x1-x2)))
        loss = loss * (2/3) + loss2 * (1/3)
        return loss

    def encode(self, x, vec_size):
        """
        Encode the x to a vector which size is vec_size
        :param x: the input tensor, which shape is [-1, img_size, img_size, 3]
        :param vec_size: the size of the semantics vector
        :return: the semantics vectors which shape is [-1, vec_size]
        """
        net = resnet.ResNet(resnet.RESNET50)
        logits = net(x, vec_size, False, 'resnet')  # logits: [-1, vec_size]
        return logits

    def decode(self, vec):
        """
        Decode the semantics vector.
        :param vec: [-1, vec_size]
        :return: [-1, img_size, img_size, 3]
        """
        net = deresnet.TransposeResNet(deresnet.RESNET50)
        y = net(vec, self.config.img_size, False, 'deresnet')  # [-1, img_size, img_size, 3]
        return y


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
