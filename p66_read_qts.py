import numpy as np

class QTS:
    def __init__(self, path):
        self.dictionary = {}
        self.chars = []
        with open(path) as file:
            lines = file.readlines()

        self.poems = []
        for line in lines:
            line = line.strip()
            if len(line) != 32:
                print(line)
            else:
                self.read_poem(line)
                self.poems.append(self.get_ids(line))

        print('Chinese chars:', len(self.chars))
        # self.poems = [self.get_ids(line.strip()) for line in lines]   # [-1, 32]
        self.num_examples = len(self.poems)
        self.pos = np.random.randint(0, self.num_examples)

    def next_batch(self, batch_size):
        next = self.pos + batch_size
        if next < self.num_examples:
            result = self.poems[self.pos: next]
        else:
            result = self.poems[self.pos:]
            next -= self.num_examples
            result.extend(self.poems[:next])
        # result: [batch_szie, 32]
        self.pos = next
        return [result]

    def read_poem(self, poem):
        for ch in poem:
            if ch not in self.dictionary:
                id = len(self.chars)
                self.dictionary[ch] = id
                self.chars.append(ch)

    def get_num_chars(self):
        return len(self.chars)

    def get_chars(self, *ids):
        result = [self.chars[id] for id in ids]
        return ''.join(result)

    def get_ids(self, s):
        return [self.dictionary[ch] for ch in s]


if __name__ == '__main__':
    qts = QTS('../samples/qts_7X4.txt')

    a = qts.get_chars(112)
    print(a)
    b = qts.get_chars(112, 334)
    print(b)

    c = [112, 334, 556, 778]
    c = qts.get_chars(*c)
    print(c)

    print(qts.get_ids(c))

    a = qts.next_batch(1)[0]
    print(qts.get_chars(*a))
