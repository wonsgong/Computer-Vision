import sys
import numpy as np 
import cv2

model = "OpenPose/pose_iter_440000.caffemodel"
config = "OpenPose/pose_deploy_linevec.prototxt"

imgNames = ["image/pose1.jpg","image/pose2.jpg","image/pose3.jpg"]

# 포즈 점 개수, 점 연결 개수, 연결 점 번호 쌍
# 사람이 한명 있다고 가정하고 진행.
nparts = 18
npairs = 17
pose_pairs = [(1, 2), (2, 3), (3, 4),  # 왼팔
              (1, 5), (5, 6), (6, 7),  # 오른팔
              (1, 8), (8, 9), (9, 10),  # 왼쪽다리
              (1, 11), (11, 12), (12, 13),  # 오른쪽다리
              (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]  # 얼굴

net = cv2.dnn.readNet(model,config)

for imgName in imgNames :
    img = cv2.imread(imgName)

    if img is None :
        print("Iamge load failed")
        continue
    
    blob = cv2.dnn.blobFromImage(img,1/255.,(368,368))
    net.setInput(blob)
    out = net.forward()

    h,w = img.shape[:2]
    points = []
    for i in range(nparts) :
        heatmap = out[0,i,:,:]

        _,conf,_,point = cv2.minMaxLoc(heatmap)

        x = int(w * point[0] / heatmap.shape[0])
        y = int(h * point[1] / heatmap.shape[1])

        points.append((x,y) if conf > 0.1 else None)
    
    for pair in pose_pairs :
        p1 = points[pair[0]]
        p2 = points[pair[1]]

        if p1 is None or p2 is None : continue

        cv2.line(img,p1,p2,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(img, p1,4,(0,0,255),-1,cv2.LINE_AA)
        cv2.circle(img, p2,4,(0,0,255),-1,cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()



