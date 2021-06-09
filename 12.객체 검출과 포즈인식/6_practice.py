import sys
import numpy as np
import cv2

def faceRecognition(net, img) :

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blob = cv2.dnn.blobFromImage(img,1/255.,(150,200))
    net.setInput(blob)
    prob = recognition_net.forward()

    _, confi, _ , maxLoc = cv2.minMaxLoc(prob)

    return maxLoc[0], confi

detection_net = cv2.dnn.readNet("face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                                "face_detector/deploy.prototxt")
recognition_net = cv2.dnn.readNet("face_rec/face_rec.pb")

cap = cv2.VideoCapture(0)

if not cap.isOpened() :
    print("Video load failed")
    sys.exit()

face_name = ['sanghuck','obama']
while True :

    ret , frame = cap.read()
    if not ret : break

    blob = cv2.dnn.blobFromImage(frame,1,(300,300),(104, 177, 123))
    detection_net.setInput(blob)
    detect = detection_net.forward()

    detect = detect[0,0,:,:]

    h,w = frame.shape[:2]

    for d in detect :
        _,_,c,x1,y1,x2,y2 = d
        if c < 0.5 : break

        x1 = int(x1 * w);   y1 = int(y1 * h)
        x2 = int(x2 * w);   y2 = int(y2 * h)

        crop = frame[y1:y2,x1:x2]
        face_idx, confi = faceRecognition(recognition_net, crop)        

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0))

        label = f"{face_name[face_idx]} : {confi:0.3f}"
        cv2.putText(frame, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27 :
        break

cap.release()
cv2.destroyAllWindows()