import sys
import numpy as np
import cv2 

model = "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = "face_detector/deploy.prototxt"

cap = cv2.VideoCapture(0)

if not cap.isOpened() :
    print("Video load failed")
    sys.exit()

net = cv2.dnn.readNet(model,config)

while True :
    ret, frame = cap.read()

    if not ret : break

    blob = cv2.dnn.blobFromImage(frame,1,(300,300),(104,177,123))
    net.setInput(blob)
    out = net.forward()
    detect = out[0,0,:,:]

    h,w = frame.shape[:2]
    for d in detect :
        _,_,c,x1,y1,x2,y2 = d

        if c < 0.5 : break

        x1 = int(x1 * w); y1 = int(y1 * h)
        x2 = int(x2 * w); y2 = int(y2 * h)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),1,cv2.LINE_AA)

        label = f"Face : {c:4.2f}"

        cv2.putText(frame, label, (x1,y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),1,cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27 :
        break
cap.release()
cv2.destroyAllWindows()





