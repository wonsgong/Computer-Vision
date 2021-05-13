import sys
import cv2
import numpy as np

def onChange(pos) :
    minR = cv2.getTrackbarPos('minRadius', 'img')
    maxR = cv2.getTrackbarPos('maxRadius', 'img')
    thre = cv2.getTrackbarPos('threshold', 'img')

    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                            param1=120,param2=thre,minRadius=minR,maxRadius=maxR)
    
    dst = src.copy()
    if circles is not None :
        for circle in circles[0] :
            cx,cy,rad = circle
            cv2.circle(dst, (cx,cy), int(rad), (0,0,255),2,cv2.LINE_AA)
    
    cv2.imshow('img',dst)

src = cv2.imread("image/dial.jpg")

if src is None:
    print("Image load failed")
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blr = cv2.GaussianBlur(gray, (0,0), 1.0)

cv2.imshow('img',src)

cv2.createTrackbar('minRadius','img',0, 100, onChange)
cv2.createTrackbar('maxRadius', 'img', 0, 150, onChange)
cv2.createTrackbar('threshold', 'img', 0, 100, onChange)
cv2.setTrackbarPos('minRadius', 'img', 10)
cv2.setTrackbarPos('maxRadius', 'img', 80)
cv2.setTrackbarPos('threshold', 'img', 40)

cv2.waitKey()
cv2.destroyAllWindows() 