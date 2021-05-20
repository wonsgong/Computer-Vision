import sys
import numpy as np
import cv2

obj = cv2.imread("image/spades.png",cv2.IMREAD_GRAYSCALE)
src = cv2.imread("image/symbols.png",cv2.IMREAD_GRAYSCALE)

if obj is None or src is None :
    print("Image load failed")
    sys.exit()

# 객체 영상 외곽선 검출
_, obj_bin = cv2.threshold(obj, 128, 255, cv2.THRESH_BINARY_INV)
obj_contours,_ = cv2.findContours(obj_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
obj_pts = obj_contours[0]

# 입력 영상 분석
_, src_bin = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for pts in contours :
    if cv2.contourArea(pts) < 1000 : continue

    rc = cv2.boundingRect(pts)
    cv2.rectangle(dst, rc, (255,0,0),1)

    dist = cv2.matchShapes(obj_pts, pts, cv2.CONTOURS_MATCH_I3, 0)

    cv2.putText(dst, str(round(dist,4)),(rc[0],rc[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),1,cv2.LINE_AA)

    if dist < 0.1:
        cv2.rectangle(dst, rc, (0,0,255),2)
    
cv2.imshow('obj', obj)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()