import numpy as np
import cv2
import sys

src = cv2.imread('image/flowers.jpg')

if src is None :
    print("Image load failed")

data = src.reshape(-1,3).astype(np.float32)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

for K in range(2,9) :
    ret, label, center = cv2.kmeans(data, K, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    dst = center[label.flatten()]
    dst = dst.reshape((src.shape))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
