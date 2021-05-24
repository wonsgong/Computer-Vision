import sys
import cv2
import numpy as np

src1 = cv2.imread("image/box.png",cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread("image/box_in_scene.png",cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None :
    print("Image load failed")
    sys.exit()


# feature = cv2.ORB_create()
feature = cv2.AKAZE_create()

kp1, desc1 = feature.detectAndCompute(src1,None)
kp2, desc2 = feature.detectAndCompute(src2,None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
matches = matcher.match(desc1,desc2)

matches.sort(key=lambda x:x.distance)

good_match = matches[:80]

pts1 = np.array([kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2).astype(np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in good_match]).reshape(-1,1,2).astype(np.float32)

homo, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

dst = cv2.drawMatches(src1, kp1, src2, kp2, good_match, None)

h,w = src1.shape[:2]
corner1 = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2).astype(np.float32)
# 점들이 행렬을 이용해서 어디로 이동하는지.
corner2 = cv2.perspectiveTransform(corner1, homo)
# drawMathces 시 1,2번 영상을 가로로 붙여서 보여주기 때문에 가로 영상만큼 쉬프트해줌
corner2 = corner2 + np.float32([w,0])

cv2.polylines(dst, [np.int32(corner2)], True, (0,0,255), 2, cv2.LINE_AA)

cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()