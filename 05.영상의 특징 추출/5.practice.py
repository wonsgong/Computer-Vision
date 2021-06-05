import sys
import cv2
import numpy as np


src = cv2.imread("image/coins2.jpg")

if src is None:
    print("Image load failed")
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blr = cv2.GaussianBlur(gray, (0,0),1)


circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=120, param2=40, minRadius=30, maxRadius=70)
dst = src.copy()
money = 0
if circles is not None :
    for circle in circles[0]:
        cx,cy,rad = circle
        cv2.circle(dst, (cx,cy),int(rad), (0,0,255),2,cv2.LINE_AA)
        
        # 원 영역 검출
        x1 = int(cx-rad)
        y1 = int(cy-rad)
        x2 = int(cx+rad)
        y2 = int(cy+rad)
        rad = int(rad)

        crop = dst[y1:y2,x1:x2,:]
        ch,cw = crop.shape[:2]

        # mask 생성
        mask = np.zeros((ch,cw),np.uint8)
        cv2.circle(mask, (cw//2,ch//2), rad, 255,-1)

        # 색상 구분
        hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        hue,_,_ = cv2.split(hsv)
        hue_shift = (hue + 40) % 180
        hue_mean = cv2.mean(hue_shift,mask)[0]
        
        won = 100
        if hue_mean < 90 :
            won = 10
        money += won

        cv2.putText(crop, str(won), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)

cv2.putText(dst, str(money)+'won', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2,0,2,cv2.LINE_AA)


cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
