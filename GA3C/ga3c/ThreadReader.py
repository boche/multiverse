from threading import Thread
import sys, time

from Config import Config


class ThreadReader(Thread):
    def __init__(self, remote_q, local_q):
        super(ThreadReader, self).__init__()
        self.setDaemon(True)

        self.remote_q = remote_q
        self.local_q = local_q
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            data = self.remote_q.get()
            self.local_q.put(data)
            # self.local_q.append(data)
