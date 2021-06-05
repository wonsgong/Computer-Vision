import sys
import cv2


src = cv2.imread("image/rose.bmp",cv2.IMREAD_GRAYSCALE)

if src is None :
    print("Image load failed!")
    sys.exit()

dst = cv2.GaussianBlur(src,(0,0),1)
dst2 = cv2.blur(src,(7,7))
cv2.imshow('src',src)
cv2.imshow('Gaussian',dst)
cv2.imshow('Mean',dst2)
cv2.waitKey()
cv2.destroyAllWindow()