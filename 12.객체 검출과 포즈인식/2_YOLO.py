import sys
import numpy as np
import cv2 

# 용량문제로 웨이트 파일은 따로 받아서
model = "YOLOv3/yolov3.weights"
config = "YOLOv3/yolov3.cfg"
namePath = "YOLOv3/coco.names"

imgNames = ["image/dog.jpg","image/person.jpg","image/sheep.jpg","image/kite.jpg"]

classNames = None
with open(namePath,"rt") as f :
    classNames = f.read().rstrip("\n").rsplit("\n")

colors = np.random.uniform(0, 255, size=(len(classNames), 3))

net = cv2.dnn.readNet(model,config)

# 출력 레이어 이름 받아오기
layerName = net.getLayerNames()
outLayer = [layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]

for imgName in imgNames :

    img = cv2.imread(imgName)

    if img is None :
        print("Image load failed")
        sys.exit()
    
    blob = cv2.dnn.blobFromImage(img,1/255.,(320,320),(0,0,0),True)
    net.setInput(blob)
    outs = net.forward(outLayer)
    
    h,w = img.shape[:2]
    boxes = []
    classIds = []
    confidences = []
    for out in outs :
        for detect in out :
            score = detect[5:]
            classId = np.argmax(score)
            confidence = score[classId]
            if confidence > 0.5 :
                cx = int(detect[0] * w); cy = int(detect[1] * h)
                bw = int(detect[2] * w); bh = int(detect[3] * h)
                x1 = int(cx - bw / 2); y1 = int(cy - bh / 2)

                boxes.append([x1,y1,bw,bh])
                confidences.append(float(confidence))
                classIds.append(int(classId))

    # 비최대 억제.
    # 40% 이상 겹치는 바운딩 박스에 대해 50% 이상 컨피던스만 선별
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in indices :
        i = i[0]
        
        className = classNames[classIds[i]]
        color = colors[classIds[i]]
        x,y,w,h = boxes[i]

        label = f"{className} : {confidences[i]:.2}"

        cv2.rectangle(img, (x,y,w,h), color, 2)
        cv2.putText(img, label, (x,y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,1,cv2.LINE_AA)
    
    t, _ = net.getPerfProfile()
    label = f"Inference time : {t*1000.0 / cv2.getTickFrequency():.2f} ms"
    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),1,cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()













