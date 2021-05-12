import sys
import numpy as np
import cv2


src = cv2.imread('image/rose.bmp')

if src is None:
    print("Image load failed")
    sys.exit()

# 보간법 비교
dst1 = cv2.resize(src,(1920,1280),interpolation=cv2.INTER_NEAREST)
dst2 = cv2.resize(src,(1920,1280))
dst3 = cv2.resize(src,(1920,1280),interpolation=cv2.INTER_CUBIC)
dst4 = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_LANCZOS4)

cv2.imshow('src',src)
cv2.imshow('dst1',dst1[500:900,400:800])
cv2.imshow('dst2', dst2[500:900, 400:800])
cv2.imshow('dst3', dst3[500:900, 400:800])
cv2.imshow('dst4', dst4[500:900, 400:800])
cv2.waitKey()
cv2.destroyAllWindows()


# 대칭
dst = cv2.flip(src,1)

cv2.imshow('src', src)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()