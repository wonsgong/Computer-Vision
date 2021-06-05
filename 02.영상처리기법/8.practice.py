import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


cap1 = cv2.VideoCapture("image/woman.mp4")
cap2 = cv2.VideoCapture("image/raining.mp4")

if not cap1.isOpened() or not cap2.isOpened() :
    print("Failed")
    sys.exit()


fps = cap1.get(cv2.CAP_PROP_FPS)
delay = round(1000/fps)

isChroma = False

while True :

    ret1, frame1 = cap1.read()
    if not ret1 : break

    if isChroma :
        ret2, frame2 = cap2.read() 
        if not ret2 : break

        frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame1_hsv, (50,150,0),(80,255,255))
        cv2.copyTo(frame2,mask,frame1)
    
    cv2.imshow('video', frame1)

    key = cv2.waitKey(delay)

    if key == ord(' '):
        isChroma = not isChroma
    
    if key == 27 :
        break

cv2.destroyAllWindows()