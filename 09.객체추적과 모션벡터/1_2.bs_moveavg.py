import sys
import numpy as np 
import cv2

cap = cv2.VideoCapture("image/PETS2000.mp4")

if not cap.isOpened() :
    print("Video load failed")
    sys.exit()

# back : unit8, fback : float32
_, back = cap.read()
back = cv2.cvtColor(back,cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0,0), 1)
fback = back.astype(np.float32)

while True :
    ret, frame = cap.read()
    if not ret : break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 1)

    # 이동 평균 계산
    fback = cv2.accumulateWeighted(gray, fback, 0.01)
    back = fback.astype(np.uint8)

    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    _,_,stats,_ = cv2.connectedComponentsWithStats(diff)

    for stat in stats[1:] :
        x,y,w,h,s = stat
        if s < 100 : continue

        cv2.rectangle(frame, (x,y,w,h), (0,0,255),2)

    cv2.imshow('frame', frame)
    cv2.imshow('back', back)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27 :
        break

cap.release()
cv2.destroyAllWindows()