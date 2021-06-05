import sys
import numpy as np
import cv2


src1 = cv2.imread('image/candies.png')
src2 = cv2.imread('image/candies2.png')

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

src_hsv1 = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)
src_hsv2 = cv2.cvtColor(src2, cv2.COLOR_BGR2HSV)

dst1 = cv2.inRange(src1, (0, 128, 0), (100, 255, 100))
dst_hsv1 = cv2.inRange(src_hsv1, (50, 150, 0), (80, 255, 255))

dst2 = cv2.inRange(src2, (0, 128, 0), (100, 255, 100))
dst_hsv2 = cv2.inRange(src_hsv2, (50, 150, 0), (80, 255, 255))

# 기존 이미지
cv2.imshow('src1', src1)
cv2.imshow('dst1', dst1)
cv2.imshow('dst_hsv1', dst_hsv1)
cv2.waitKey()
cv2.destroyAllWindows()

# 어두워진 이미지
cv2.imshow('sr2', src2)
cv2.imshow('dst2', dst2)
cv2.imshow('dst_hsv2', dst_hsv2)
cv2.waitKey()



cv2.destroyAllWindows()
