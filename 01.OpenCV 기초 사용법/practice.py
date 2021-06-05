import sys
import cv2
import numpy as np

cap1 = cv2.VideoCapture("image/video1.mp4")
cap2 = cv2.VideoCapture("image/video2.mp4")


if not cap1.isOpened() or not cap2.isOpened() :
    print("Failed")
    sys.exit()

frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = round(cap1.get(cv2.CAP_PROP_FPS))
delay = round(1000 / fps)
effect_frame = frame_cnt1 - fps

cv2.namedWindow('Video')

for i in range(frame_cnt1 - effect_frame) :

    ret, frame = cap1.read()

    if not ret : 
        print("Failed")
        sys.exit()
    
    cv2.imshow('Video',frame)
    cv2.waitKey(delay)


for i in range(effect_frame) :
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2 :
        print("failed")
        sys.exit()

    frame = np.zeros((h,w,3),np.uint8) 
    dx = effect_frame * i
    frame[:,dx:,:] = frame1[:,dx:,:]
    frame[:,:dx,:] = frame2[:,:dx,:]

    cv2.imshow('Video',frame)
    cv2.waitKey(delay)

for i in range(effect_frame,frame_cnt2) :
    ret, frame = cap2.read()

    if not ret : 
        print("failed")
        sys.exit()
    
    cv2.imshow('Video',frame)
    cv2.waitKey(delay)


cap1.release()
cap2.release()
cv2.destroyAllWindows()
