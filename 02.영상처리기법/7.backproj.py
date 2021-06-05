import sys
import numpy as np
import cv2


# 입력 영상에서 ROI를 지정하고, 히스토그램 계산
# 마스크 영상을 가지고 계산도 가능.
src = cv2.imread('image/cropland.png')

if src is None:
    print('Image load failed!')
    sys.exit()

# ROI 지정 함수
x, y, w, h = cv2.selectROI(src)

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
crop = src_ycrcb[y:y+h, x:x+w]

hist = cv2.calcHist([crop], [1,2], None, [256,256], [0,256,0,256])

# 히스토그램을 log 스케일 해줌. 
hist_norm = cv2.normalize(cv2.log(hist+1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 입력 영상 전체에 대해 히스토그램 역투영
backproj = cv2.calcBackProject([src_ycrcb], [1,2], hist, [0,256,0,256], 1)
dst = cv2.copyTo(src, backproj)

cv2.imshow('backproj', backproj)
cv2.imshow('hist_norm', hist_norm)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
