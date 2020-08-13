import tensorflow as tf
from p45_celeba import CelebA
from p48_BufferDS import BufferDS
import p50_framework as myf
import p51_face_recog as p51


class MyConfig(p51.MyConfig):
    def __init__(self, persons):
        super(MyConfig, self).__init__(persons)
        self.scale = 30  # gamma

    def get_name(self):
        return 'p63'

    def get_sub_tensors(self, gpu_id):
        return MySubTensors(self)

    def test(self):
        app = self.get_app()
        with app:
            pass


class MySubTensors(p51.MySubTensors):
    def get_logits(self, vector):
        # vector: [-1, vec_size]
        vector = tf.nn.l2_normalize(vector, axis=1)
        w = tf.get_variable('std_vector', [vector.shape[-1].value, self.config.persons], tf.float32)
        w = tf.nn.l2_normalize(w, axis=0)
        logits = tf.matmul(vector, w)   # [-1, persons]
        return logits * self.config.scale


if __name__ == '__main__':
    path_img = '../samples/celeba/Img/img_align_celeba.zip'
    path_ann = '../samples/celeba/Anno/identity_CelebA.txt'
    path_bbox = '../samples/celeba/Anno/list_bbox_celeba.txt'
    celeba = CelebA(path_img, path_ann, path_bbox)
    cfg = MyConfig(celeba.persons)
    ds = BufferDS(cfg.buffer_size, celeba, cfg.batch_size)
    cfg.ds = ds

    cfg.from_cmd()
