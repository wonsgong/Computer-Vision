import sys
import numpy as np
import cv2

def decode(rbox, scores, confThre) :
    detections = []
    confidences = []
    
    sh,sw = scores.shape

    for y in range(sh) :
        for x in range(sw) :

            score = scores[y][x]
            angle = rbox[4][y][x]
            x0 = rbox[0][y][x]; x1 = rbox[1][y][x]
            x2 = rbox[2][y][x]; x3 = rbox[3][y][x]

            if score < confThre : continue

            # feature map 크기인 320 * 320으로 확대.(현재 80 * 80)
            offsetX = x * 4.0; offsetY = y * 4.0
            
            # (offsetX, offsetY) 에서 회전된 사각형 정보 추출
            cosA = np.cos(angle)
            sinA = np.sin(angle)
            h = x0 + x2
            w = x1 + x3

            # 회전된 사각형의 한쪽 모서리 점 좌표 계산
            offset = ([offsetX + cosA * x1 + sinA * x2, offsetY - sinA * x1 + cosA * x2])
            
            # 회전된 사각형의 대각선의 위치한 두 모서리 점 좌표 계산
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = ((p1[0]+p3[0])/2,(p1[1]+p3[1])/2)

            detections.append((center,(w,h),-1*angle * 180. / np.pi))
            confidences.append(float(score))

    return [detections,confidences]


model = 'EAST/frozen_east_text_detection.pb'
confThre = 0.5
nmsThre = 0.4

imgNames = ['image/road_closed.jpg', 'image/patient.jpg', 'image/copy_center.jpg']

net = cv2.dnn.readNet(model)

layerNames = net.getLayerNames()
outLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# outLayers = ['feature_fusion/concat_3','feature_fusion/Conv_7/Sigmoid']

for imgName in imgNames :
    img = cv2.imread(imgName)

    if img is None :
        print("Image load failed")
        continue

    blob = cv2.dnn.blobFromImage(img,1,(320,320),(123.68,116.78,103.94),True)
    net.setInput(blob)
    rboxes, scores = net.forward(outLayers)

    # score > confThre 인 rbox 정보를 RotatedRect 형식으로 변환해서 반환.
    [boxes, confidences] = decode(rboxes[0,:,:,:], scores[0,0,:,:], confThre)
 

    # 비최대 억제
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThre, nmsThre)

    rh = img.shape[0] / 320
    rw = img.shape[1] / 320
    
    for i in indices :
        # 회전된 사각형의 네 모서리 점 좌표 계산 
        vertices = cv2.boxPoints(boxes[i[0]])

        for j in range(4) :
            vertices[j][0] *= rw
            vertices[j][1] *= rh

        for j in range(4) :
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j+1) % 4][0], vertices[(j+1) % 4][1])
            cv2.line(img, p1, p2, (0,0,255),2,cv2.LINE_AA)
    

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
