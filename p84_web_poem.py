from tornado import web, ioloop
from p69_poem_multi_dropout import MyConfig


config = None
config.keep_prob = None
poem_app = None


class PoemHandler(web.RequestHandler):

    def post(self):
        self.get()

    def get(self):
        head = self.get_argument('head')
        global poem_app
        poem = poem_app.make_poem(head)
        self.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>唐诗创作</title>
</head>
<body>
    <h1 align="center">唐诗创作</h1>
    <div style="text-align:center">用户输入：{head}</div>
    <div style="text-align:center">生成的唐诗：</div>
    <div style="text-align:center; font-size:50px">{poem}</div>
    <a href="/static/poem.html">返回首页</a>
</body>
</html>
        """.format(head=head, poem=poem))


if __name__ == '__main__':
    config = MyConfig()
    config.keep_prob = 1.0
    poem_app = config.get_app()
    app = web.Application([
        ('/static/(.*)', web.StaticFileHandler, {'path': './html/', 'default_filename':'poem.html'}),
        ('/poem', PoemHandler)
    ])
    app.listen(5678)

    current = ioloop.IOLoop.current()
    current.start()
