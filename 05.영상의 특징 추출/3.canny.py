import sys
import cv2
import numpy as np

src = cv2.imread("image/building.jpg", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

# thre 적절하게 조절하면된다. 
dst = cv2.Canny(src,50,150)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()