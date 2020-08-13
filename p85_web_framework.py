import traceback

from tornado import web, ioloop
import os
import p74_framework as myf
import threading

_configs = {}


def list_pys(path):
    if not path.endswith(os.sep):  # "/" ./ ..
        path += os.sep
    result = []
    for file_name in os.listdir(path):
        if file_name.startswith('p') and file_name.endswith(".py"):
            name = file_name[:-3]
            config_type = get_config_type(name)
            if config_type is not None:
                result.append(name)
    return sorted(result)


def get_config_type(name):
    try:
        exec("import %s as prog" % name)
        for member_name in eval("dir(prog)"):
            member = eval("prog.%s" % member_name)
            if member != myf.Config and type(member) == type and issubclass(member, myf.Config):
                return member
        return None
    except:
        # traceback.print_exc()
        print("None")
        return None


def get_config(name):
    global _configs
    if name not in _configs:
        config = get_config_type(name)()
        _configs[name] = config
    return _configs[name]


class HomeHandler(web.RequestHandler):
    def post(self):
        self.get()

    def get(self):
        pys = []
        for py in list_pys('.'):
            line = '<tr>' \
                   '    <td>' + py + \
                   '    </td>' \
                   '    <td>' \
                   '        <a href="/start?py={py}">Start</a>' \
                   '    </td>' \
                   '    <td><a href="start?new_model=on&py={py}">new_start</td>' \
                   '    <td>' \
                   '        <a href="/stop?py={py}">Stop</a>' \
                   '    </td>' \
                   '<td><a href="/test?py={py}">Test</a>' \
                   '<td><a href="edit_config?py={py}">Configuration</a>' \
                   '</tr>' \
                       .format(py=py)
            pys.append(line)
        pys = ''.join(pys)

        self.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>我的AI模型</title>
            </head>
            <body>
                <h1 align="center">我的AI模型</h1>
                
                <table align="center" border="1" cellspacing="0">
                    <tr><td colspan="5" align="right">
                            Cuda ="{gpu}"
                    </td></tr>
                    {pys}
                </table>
            </body>
            </html>
        """.format(pys=pys, gpu=os.getenv("CUDA_VISBLE_DEVICES", "0")))


class StartHandler(web.RequestHandler):
    def post(self):
        self.get()

    def get(self):
        py = self.get_argument('py')
        new_model = self.get_argument("new_model", None)
        new_model = new_model is not None
        self.write('start training ' + py)
        exec("import %s as prog" % py)

        def start_training():
            config = get_config(py)
            config.new_model = new_model
            config.train()

        th = threading.Thread(daemon=True, target=start_training)
        th.start()


class StopHandler(web.RequestHandler):
    def post(self):
        self.get()

    def get(self):
        py = self.get_argument('py')
        self.write('stop training ' + py)
        config = get_config(py)
        config.stopped = True


class EditConfigHandler(web.RequestHandler):
    def post(self):
        self.get()

    def get(self):
        py = self.get_argument("py")
        lr = self.get_argument("lr", None)
        epoches = self.get_argument("epoches", None)
        batch_size = self.get_argument("batch_size", None)
        config = get_config(py)
        if lr is not None:
            config.lr = float(lr)

        if epoches is not None:
            config.epoches = int(epoches)

        if batch_size is not None:
            config.batch_size = int(batch_size)

        self.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>更改</title>
        </head>
        <body>
            <h1 align="center">更改</h1>
            <form action="edit_config">
            <input type="hidden" name="py" value="{py}"/>
            <table align="center" border="1" cellspacing="0">
                <tr><td>batch_size:</td><td><input name ="batch_size" value="{batch_size}"></td></tr>
                <tr><td>epoches :</td><td><input name="epoches" value="{epoches}"></td></tr>
                <tr><td>lr:</td><td><input name="lr" value="{lr}"></td></tr>
                <tr><td colspan="2" align="center"><input type="submit" value="Modify"></td></tr>
            </table>
            <div style="text-alignment:right"><a href="/home">home</a>
        </body>
        </html>
                """.format(py=py, lr=config.lr, batch_size=config.batch_size, epoches=config.epoches))


class GPUHandler(web.RequestHandler):
    def post(self):
        self.get()

    def get(self):
        gpu = self.get_argument("gpu", "0")
        os.putenv("CUDA_VISBLE_DEVICES")
        self.write("SET TO" + gpu)


class TestHandler(web.RequestHandler):
    def post(self):
        self.get()

    def get(self):
        py = self.get_argument("py")
        self.write("Test on %s starts." % py)
        config = get_config(py)
        print("Test starts.")
        config.test()


if __name__ == '__main__':
    app = web.Application([
        ('/static/(.*)', web.StaticFileHandler, {'path': './html/', 'default_filename': 'poem.html'}),
        ('/home', HomeHandler),
        ('/start', StartHandler),
        ('/stop', StopHandler),
        ('/test', TestHandler),
        ('/edit_config', EditConfigHandler),
        ('./gpu', GPUHandler)
    ])
    app.listen(5678)

    current = ioloop.IOLoop.current()
    current.start()
