import sys
import numpy as np
import math
import cv2

src = cv2.imread('image/tekapo.bmp')

if src is None:
    print("Image load failed")
    sys.exit()

# 좌측 상단 기준 회전.
rad = 20 * math.pi / 180
aff = np.array([[math.cos(rad),math.sin(rad),0],
                [-math.sin(rad),math.cos(rad),0]],dtype=np.float32)
dst = cv2.warpAffine(src, aff, (0,0))


# 중심 기준 회전
h,w = src.shape[:2]
cp = (w//2,h//2)
aff = cv2.getRotationMatrix2D(cp, 20, 1)
dst2 = cv2.warpAffine(src, aff, (0,0))


cv2.imshow('src',src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()