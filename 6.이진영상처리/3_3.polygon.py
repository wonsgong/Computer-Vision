import sys
import numpy as np
import cv2


def setLabel(img, pts, label) :
    (x,y,w,h) = cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x+w, y+h)
    cv2.rectangle(img, pt1, pt2, (0,0,255),1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),lineType=cv2.LINE_AA)


src = cv2.imread('image/polygon.bmp')

if src is None:
    print("Image load failed")
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, src_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for pts in contours :

    if cv2.contourArea(pts) < 400 : continue
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)
    vtc = len(approx)

    if vtc == 3 :
        setLabel(src,pts,'TRI')
    elif vtc == 4:
        setLabel(src,pts,"RECT")
    else :
        lenth = cv2.arcLength(pts, True)
        area = cv2.contourArea(pts)
        ratio = 4. * np.pi * area / (lenth * lenth)

        if ratio > 0.85 :
            setLabel(src, pts, "Circle")



cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
