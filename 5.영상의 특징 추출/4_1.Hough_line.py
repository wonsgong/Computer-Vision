import sys
import cv2
import numpy as np

src = cv2.imread("image/building.jpg", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()


edge = cv2.Canny(src,50,150)
# 파라미터를 잘 줘야 된다.
lines = cv2.HoughLinesP(edge, 1.0, np.pi / 180, 160,minLineLength=50,maxLineGap=5)

dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
if lines is not None :
    for line in lines :
        # shape = (N,1,4)
        pt1 = (line[0][0],line[0][1])
        pt2 = (line[0][2], line[0][3])
        cv2.line(dst,pt1,pt2,(0,0,255),2,cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
