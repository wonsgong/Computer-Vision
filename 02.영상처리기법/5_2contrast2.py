import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 히스토그램 평활화

src = cv2.imread('image/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 그레이스케일 영상의 히스토그램 평활화
dst1 = cv2.equalizeHist(src)

# 그레이스케일 영상의 히스토그램 스트레칭
dst2 = cv2.normalize(src, None,0,255,cv2.NORM_MINMAX)


# 히스토그램 비교
srchist = cv2.calcHist([src], [0], None, [256], [0,256])
dst1hist = cv2.calcHist([dst1], [0], None, [256], [0,256])
dst2hist = cv2.calcHist([dst2], [0], None, [256], [0,256])


cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey(1)

plt.subplot2grid((1,3), (0,0)), plt.plot(srchist), plt.title("src")
plt.subplot2grid((1,3), (0,1)), plt.plot(dst1hist), plt.title("dst1")
plt.subplot2grid((1,3), (0,2)), plt.plot(dst2hist), plt.title("dst2")
plt.show()


cv2.destroyAllWindows()

# 컬러 영상의 히스토그램 평활화
src = cv2.imread('field.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
planes = cv2.split(src_ycrcb)
planes[0] = cv2.equalizeHist(planes[0])

dst = cv2.merge(planes)
dst = cv2.cvtColor(dst, cv2.COLOR_YCrCb2BGR)

cv2.imshow('src', src)
cv2.imshow('dst',dst)
cv2.waitKey()

cv2.destroyAllWindows()