from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())

print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

print("[INFO] starting the FPS timer...")
fps = FPS().start()

while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    cv2.putText(
        img=frame,
        text="Queue size: {}".format(fvs.Q.qsize()),
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 0),
        thickness=2
    )

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
fvs.stop()