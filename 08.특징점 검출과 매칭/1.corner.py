import sys
import cv2
import numpy as np

src = cv2.imread("image/building.jpg",cv2.IMREAD_GRAYSCALE)

if src is None :
    print("Image load failed")
    sys.exit()

tm = cv2.TickMeter()

tm.start()

corners = cv2.goodFeaturesToTrack(src, 400, 0.01, 10)

tm.stop()
print("GFTT : {}ms".format(tm.getTimeMilli()))

dst1 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

if corners is not  None:
    for corner in corners :
        pt = (int(corner[0][0]),int(corner[0][1]))
        cv2.circle(dst1, pt, 5, (0,0,255),2,cv2.LINE_AA)

tm.reset()
tm.start()

fast = cv2.FastFeatureDetector_create(60)
keypoints = fast.detect(src)

tm.stop()
print("FAST : {}ms".format(tm.getTimeMilli()))

dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
for kp in keypoints :
    pt = (int(kp.pt[0]), int(kp.pt[1]))
    cv2.circle(dst2,pt,5,(0,0,255),2,cv2.LINE_AA)



cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
