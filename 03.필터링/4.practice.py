import sys
import numpy as np
import cv2

def cartoon_filter(img) :
    h,w = img.shape[:2]
    img_resize = cv2.resize(img,(w//2,h//2))

    blr = cv2.bilateralFilter(img_resize,-1,10,5)
    edge = 255 - cv2.Canny(img_resize,80,120)
    edge = cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)

    dst = cv2.bitwise_and(blr,edge)
    dst = cv2.resize(dst,(w,h),interpolation=cv2.INTER_NEAREST)

    return dst 

def pencil_sketch(img) :
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray,(0,0),2)
    dst = cv2.divide(gray,blr,scale=255)

    dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)

    return dst 

cap = cv2.VideoCapture(0)

if not cap.isOpened() : 
    print("Video load failed!")
    sys.exit()


ch = 0
while True :
    
    ret , frame = cap.read()

    if not ret : 
        print("Read failed!")
        sys.exit()
    
    if ch == 1 :
        frame = cartoon_filter(frame)
    if ch == 2 :
        frame = pencil_sketch(frame)
        
    cv2.imshow('Video',frame)

    key = cv2.waitKey(1)
    if key == ord(' '):
        ch += 1
        if ch == 3 : ch = 0
    
    if key == 27 :
        break

cap.release()
cv2.destroyAllWindows()