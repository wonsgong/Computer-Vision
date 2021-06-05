import sys
import numpy as np 
import cv2

cap = cv2.VideoCapture("image/vtest.mp4")

if not cap.isOpened() :
    print("Video load falied")
    sys.exit()

_, frame1 = cap.read()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
# hsv[...,1] = 255 와 같다. 
hsv[:,:,1] = 255
while True :
    ret, frame2 = cap.read()
    if not ret : break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1,gray2,None,0.5, 3, 13, 3, 5, 1.1, 0)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

    hsv[:,:, 0] = ang*180/np.pi/2
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', frame2)
    cv2.imshow('bgr', bgr)

    if cv2.waitKey(30) == 27 :
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()
