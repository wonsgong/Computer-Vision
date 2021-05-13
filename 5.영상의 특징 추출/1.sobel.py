import sys
import cv2
import numpy as np

src = cv2.imread("image/lenna.bmp",cv2.IMREAD_GRAYSCALE)

if src is None :
    print("Image load failed")
    sys.exit()


# 소벨 마스크 생성
mask = np.array([[-1,0,1],
                 [-1,0,1],
                 [-1,0,1]],dtype=np.float32)

dst = cv2.filter2D(src, -1, mask,delta=128)


# 소벨 함수 사용
dx = cv2.Sobel(src, -1, 1,0,delta=128)
dy = cv2.Sobel(src, -1, 0,1,delta=128)

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('dx',dx)
cv2.imshow('dy',dy)

cv2.waitKey()
cv2.destroyAllWindows()