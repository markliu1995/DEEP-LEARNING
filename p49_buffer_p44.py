from p44_CVAE_mnist_gpus import MyConfig, read_data_sets, MyDS, predict
from p43_framework import App
from p48_BufferDS import BufferDS


def my_ds(ds, cfg):
    ds = MyDS(ds, cfg)
    return BufferDS(10, ds, cfg.batch_size)


if __name__ == '__main__':
    class Config(MyConfig):
        def get_name(self):
            return 'p49'

    cfg = Config()
    cfg.from_cmd()
    print('-' * 100)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)
    app = App(cfg)
    with app:
        # app.train(my_ds(dss.train, cfg), my_ds(dss.validation, cfg))
        predict(app, 500, cfg.img_path, cfg.cols)
