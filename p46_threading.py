import threading
import time

n = 0
lock = threading.Lock()


def add():
    global n, lock
    for _ in range(500000):
        # lock.acquire()
        with lock:
            n += 1
        # lock.release()

        # lock.acquire()
        # try:
        #     n += 1
        # finally:
        #     lock.release()

def sub():
    global n
    for _ in range(500000):
        # lock.acquire()
        with lock:
            n -= 1
        # lock.release()


if __name__ == '__main__':
    # main thread
    thread_add = threading.Thread(target=add)
    thread_sub = threading.Thread(target=sub)
    thread_add.start()
    thread_sub.start()

    thread_add.join()
    thread_sub.join()
    print('n =', n, flush=True)
