import sys
import numpy as np 
import cv2

cap = cv2.VideoCapture("image/PETS2000.mp4")

if not cap.isOpened() :
    print("Video load failed")
    sys.exit()

# bs = cv2.createBackgroundSubtractorMOG2()
bs = cv2.createBackgroundSubtractorKNN()
while True :
    ret, frame = cap.read()

    if not ret : break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = bs.apply(gray)
    back = bs.getBackgroundImage()

    _,_,stats,_ = cv2.connectedComponentsWithStats(fgmask)

    for stat in stats[1:] :
        x,y,w,h,s = stat
        if s < 100 : continue

        cv2.rectangle(frame, (x,y,w,h), (0,0,255),2)
    
    cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('back',back)

    if cv2.waitKey(30) == 27 :
        break

cap.release()
cv2.destroyAllWindows()