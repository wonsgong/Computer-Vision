import sys
import numpy as np
import cv2

# 맥에서 영상이 왜 안열리지...
cap = cv2.VideoCapture("image/vtest.avi")

if not cap.isOpened() :
    print("Video open failed")
    sys.exit()


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
delay = int(cap.get(cv2.CAP_PROP_FPS) / 1000)
while True :
    ret, frame = cap.read() 

    if not ret :
        break

    people,_ = hog.detectMultiScale(frame)

    for rc in people :
        cv2.rectangle(frame,rc,(0,0,255),2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(delay) == 27 :
        break

cap.release()
cv2.destroyAllWindows()