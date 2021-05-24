import sys
import cv2
import numpy as np

src1 = cv2.imread("image/graf1.png",cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread("image/graf3.png",cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None :
    print("Image load failed")
    sys.exit()



# feature = cv2.KAZE_create()
# feature = cv2.AKAZE_create()
feature = cv2.ORB_create()

kp1 = feature.detect(src1)
_, desc1 = feature.compute(src1,kp1)

kp2, desc2 = feature.detectAndCompute(src2,None)


dst1 = cv2.drawKeypoints(src1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst2 = cv2.drawKeypoints(src2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()