class DialogDS:
    def __init__(self, samples):
        self.sub_samples = [sample.sub_samples() for sample in samples]
        self.index = 0

    def next_batch(self, batch_size):
        result = []
        for _ in range(batch_size):
            result.append(next(self.sub_samples[self.index]))
            self.index = (self.index + 1) % len(self.sub_samples)
        return result


X_ASK = 0
Y_ASK = 1

X = 0
Y = 1

STATE_0 = 0
STATE_X_ASK = 1
STATE_Y_ASK = 2


class Sample:
    def __init__(self, background, dialogs):
        self.background = background
        self.dialogs = dialogs

    def sub_samples(self):
        mode = X_ASK
        while True:
            yield from self.state_transfer(mode)
            mode = 1 - mode

    def state_transfer(self, mode):
        background = self.background
        state = STATE_0
        x_question = None
        y_question = None
        for d in self.dialogs:
            if state == STATE_0:
                if d.who == mode:
                    if d.question:
                        state = STATE_X_ASK
                        x_question = d.what
                    else:
                        background += convert(d)
                else:
                    if d.question:
                        y_question = d.what
                        state = STATE_Y_ASK
                    else:
                        yield background, None, d.what

            elif state == STATE_X_ASK:
                if d.who == mode:
                    if d.question:
                        yield background, x_question, None
                        background += convert(d)
                        x_question = d.what
                    else:
                        background += convert(d)
                        state = STATE_0
                else:
                    if d.question:
                        yield background, x_question, None
                        state = STATE_Y_ASK
                    else:
                        yield background, x_question, d.what
                        state = STATE_0
            else:
                yield background, None, y_question
                if d.who == mode:
                    if d.question:
                        x_question = d.what
                        state = STATE_X_ASK
                    else:
                        background += convert(d)
                        state = STATE_0
                else:
                    if d.question:
                        y_question = d.what

                    elif not d.question:
                        yield background, None, d.what
                        state = STATE_0
        if state == STATE_X_ASK:
            yield background, x_question, None


def convert(d):
    return d.what


class Dialog:
    def __init__(self, who, what, question=None):
        self.who = who
        self.what = what
        self.question = self.what.endswith('?') if question is None else question


if __name__ == "__main__":
    Dialog = [Dialog(X, "aaaaaa?"),
              Dialog(Y, "aaaaaad"),
              Dialog(X, "bbbbbbbbbbb?"),
              Dialog(Y, "aaaasdsdsdsd")
              ]
    sample = Sample("ABCDE", Dialog)
    ds = DialogDS([sample])
    for values in ds.next_batch(8):
        print(values)
