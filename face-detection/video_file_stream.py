from threading import Thread
import sys
import cv2

if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

class VideoFileStream:
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)
    
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return    
            
            if not self.queue.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stop()
                    return

                self.queue.put(frame)
    
    def read(self):
        return self.queue.get()

    def more(self):
        return self.queue.qsize > 0

    def stop(self):
        self.stopped = True