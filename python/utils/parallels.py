import time
import threading


class Worker(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            item = self.queue.get()

            item.do_job()

            self.queue.task_done()


class LoggerTask(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.__terminate = False

    def terminate(self):
        self.__terminate = True

    def run(self):
        attempts = 0

        while self.queue.qsize() == 0:
            if self.__terminate:
                print("[Parallels][Logger] Terminate")
                return
            attempts += 1
            if attempts > 10:
                print("[Parallels][Logger] Query is empty. Break logger. Attempts: {attempts}".
                      format(attempts=attempts))
                return
            print("[Parallels][Logger] Task in query zero. Try after 5 seconds. Attempts: {attempts}".
                  format(attempts=attempts))
            time.sleep(5)

        while self.queue.qsize() > 0:
            if self.__terminate:
                print("[Parallels][Logger] Terminate")
                return
            print("[Parallels][Logger] Task in query {count}".format(count=self.queue.qsize()))
            time.sleep(5)
