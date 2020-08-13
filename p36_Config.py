import argparse


class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 2000
        self.batch_size = 50
        self.logdir = '../logs/{name}'.format(name=self.get_name())
        self.save_path = '../models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.new_model = False

    def get_name(self):
        raise Exception('get_name() is not redefined.')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s=%s' % (attr, attrs[attr]) for attr in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, str, bool):
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            if type(value) == bool:
                parser.add_argument('--' + attr, default=value, help='Default to %s' % value,
                                    action=('store_%s' % (not value)).lower())
            else:
                t = str if value is None else type(value)
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            if hasattr(a, attr):
                setattr(self, attr, getattr(a, attr))


if __name__ == '__main__':
    class MyConfig(Config):
        def __init__(self):
            super(MyConfig, self).__init__()
            self.abc = 12.34
            self.xyz = True

        def get_name(self):
            return 'my_app'

    cfg = MyConfig()
    print(cfg)
    print('-' * 50)
    cfg.from_cmd()
    print(cfg)
