from tornado import web, ioloop
from p69_poem_multi_dropout import MyConfig

config = MyConfig()
config.keep_prob = 1.0
poemapp =config.get_app()


class PoemHandler(web.RequestHandler):
    def get(self):
        head = self.get_argument("head")
        poem = poemapp.make_poem(head)
        self.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Tang dynasty</title>
        </head>
        <body>
            <h1 align="center">
                Chinese Poem Forever!!!
            </h1>
            <div style="text_align:center">User input:{head}</div>
            <div style="text-align: center">生成的诗歌<div>
            <div>{poem}<div>
            <h2><a href="./poem/poem.html">Return to the index page.</a><h2>
            <p align="center">
            </p>
        </body>
        </html>
        """.format(head=head, poem=poem))

if __name__ == "__main__":
    app = web.Application([
        ("/poem/(.*)", web.StaticFileHandler, {"path": "./html/"}),
        ("/hello/(.*)", web.StaticFileHandler, {"path": "./html/"}),
        ("/def", PoemHandler)
    ])
    app.listen(9045)
    current = ioloop.IOLoop.current()
    current.start()