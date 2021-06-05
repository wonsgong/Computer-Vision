import sys
import numpy as np 
import cv2

cap = cv2.VideoCapture("image/camshift.mp4")

if not cap.isOpened() :
    print("Video load failed")
    sys.exit()

# 미리 주어진 ROI 좌표
x,y,w,h = 135,220,100,100
rc = (x,y,w,h)

_, roi = cap.read()
roi = roi[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# HS Hist 계산
channels = [0,1]
ranges = [0,180,0,256]
hist = cv2.calcHist([roi_hsv], channels, None, [48,64], ranges)

# term_criteria 
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT,10,1)

while True :
    ret, frame = cap.read() 
    if not ret : break

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    backproj = cv2.calcBackProject([hsv], channels, hist, ranges, 1)

    _,rc = cv2.meanShift(backproj, rc, term_crit)

    cv2.rectangle(frame, rc, (0,0,255),2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(60) == 27 :
        break

cap.release()
cv2.destroyAllWindows()