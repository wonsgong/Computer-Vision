import sys
import numpy as np 
import cv2

cap = cv2.VideoCapture("image/PETS2000.mp4")

if not cap.isOpened() :
    print("Video load failed")
    sys.exit()

_ , back = cap.read()
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0,0), 1)

while True :
    ret, frame = cap.read()
    if not ret :
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 1)

    # 차영상 구하기 + 이진화
    diff = cv2.absdiff(back, gray)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 바운딩 박스 표시
    _, _, stats, _ = cv2.connectedComponentsWithStats(diff)
    for stat in stats[1:] :
        x,y,w,h,s = stat
        if s < 100 : continue
        cv2.rectangle(frame,(x,y,w,h), (0,0,255), 2) 

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27 :
        break

cap.release()
cv2.destroyAllWindows()