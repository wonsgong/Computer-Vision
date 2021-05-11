import sys
import cv2


src = cv2.imread("image/noise.bmp", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

dst = cv2.medianBlur(src,5)

cv2.imshow("src",src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindow()
