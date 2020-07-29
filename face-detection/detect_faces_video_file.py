import numpy as np  
from imutils.video import FileVideoStream
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, 
    help="path to Caffe 'Deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video file stream...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

while fvs.more():
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)

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
        (start_x, start_y, end_x, end_y) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10

        cv2.rectangle(
            img=frame,
            pt1=(start_x, start_y),
            pt2=(end_x, end_y),
            color=(0, 0, 255),
            thickness=2
        )

        cv2.putText(
            img=frame,
            text=text,
            org=(start_x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.47,
            color=(0, 0, 225),
            thickness=2
        )
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] closing windows and cleaning up...")
cv2.destroyAllWindows()
fvs.stop()
print("[INFO] end of program.")