import sys
import numpy as np
import cv2


src = cv2.imread('image/tekapo.bmp')

if src is None:
    print("Image load failed")
    sys.exit()

h,w = src.shape[:2]
mapy, mapx = np.indices((h,w),dtype=np.float32)

mapy = mapy + 10 * np.sin(mapx / 32)

dst = cv2.remap(src,mapx,mapy,cv2.INTER_LINEAR)

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()
