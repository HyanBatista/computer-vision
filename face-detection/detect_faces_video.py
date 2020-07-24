import numpy as np  
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, 
    help="path to Caffe 'Deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(
            img=frame,
            pt1=(startX, startY),
            pt2=(endX, endY),
            color=(0, 0, 255),
            thickness=2
        )
        cv2.putText(
            img=frame,
            text=text,
            org=(startX, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.47,
            color=(0, 0, 225),
            thickness=2
        )

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()