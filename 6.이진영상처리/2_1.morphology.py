import cv2
import numpy as np
import sys

src = cv2.imread('image/circuit.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

# 구조 요소 생성
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))

# 침식 & 팽창
dst1 = cv2.erode(src, se)
dst2 = cv2.dilate(src, se)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()