import sys
import cv2
import numpy as np

# 기준 영상
src = cv2.imread('image/korea.jpg',cv2.IMREAD_GRAYSCALE)

if src is None :
    print("Image load failed")
    sys.exit()

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('image/korea.mp4')
if not cap1.isOpened() or not cap2.isOpened() :
    print("Video open falied")
    sys.exit()

# AKAZE 생성
detector = cv2.AKAZE_create()

# 기준 영상에서 특징점 검출 및 기술자 생성
kp1, desc1 = detector.detectAndCompute(src,None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
delay = int(cap1.get(cv2.CAP_PROP_FPS) // 1000)
while True :
    ret1, frame1 = cap1.read()

    if not ret1 :
        break
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = detector.detectAndCompute(gray,None)

    if len(kp2) > 100 :
        matches = matcher.match(desc1,desc2)

        matches = sorted(matches,key=lambda x:x.distance)
        matches = matches[:80]

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2).astype(np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2).astype(np.float32)

        homo, inlier = cv2.findHomography(pts1,pts2,cv2.RANSAC)

        inlier_cnt = cv2.countNonZero(inlier)

        if inlier_cnt > 20 :

            ret2, frame2 = cap2.read()

            if not ret2 :
                break

            h,w = frame1.shape[:2]

            video_warp = cv2.warpPerspective(frame2, homo, (w,h))

            white = np.full(frame2.shape[:2], 255, np.uint8)
            white = cv2.warpPerspective(white, homo, (w,h))

            cv2.copyTo(video_warp,white,frame1)
    
    cv2.imshow('frame', frame1)
    key = cv2.waitKey(1)
    if key == 27 :
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()