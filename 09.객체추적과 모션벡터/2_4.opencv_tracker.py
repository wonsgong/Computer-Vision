import sys
import numpy as np
import cv2

cap = cv2.VideoCapture('image/tracking2.mp4')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 트래커 객체 생성

# Kernelized Correlation Filters
# tracker = cv2.TrackerKCF_create()

# Minimum Output Sum of Squared Error
#tracker = cv2.TrackerMOSSE_create()

# Discriminative Correlation Filter with Channel and Spatial Reliability
tracker = cv2.TrackerCSRT_create()

_,frame = cap.read()

roi = cv2.selectROI('frame', frame)
tracker.init(frame,roi)

while True :
    ret, frame = cap.read() 
    if not ret : continue

    _,roi = tracker.update(frame)

    roi = tuple([int(_) for _ in roi])

    cv2.rectangle(frame, roi, (0,0,255),2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27 :
        break
cap.release()
cv2.destroyAllWindows()