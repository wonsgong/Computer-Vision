import sys
import numpy as np
import cv2 

def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{classNames[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                  (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 255, 1, cv2.LINE_AA)

model = 'Mask-RCNN/frozen_inference_graph.pb'
config = 'Mask-RCNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
namePath = 'Mask-RCNN/coco_90.names'

imgNames = ["image/dog.jpg","image/traffic.jpg","image/sheep.jpg"]

classNames = None
with open(namePath,"rt") as f :
    classNames = f.read().rstrip("\n").rsplit("\n")
colors = np.random.uniform(0, 255, size=(len(classNames), 3))


net = cv2.dnn.readNet(model,config)

for imgName in imgNames :

    img = cv2.imread(imgName)

    if img is None : 
        print("Image load failed")
        sys.exit()

    blob = cv2.dnn.blobFromImage(img,swapRB=True)
    net.setInput(blob)
    boxes , masks = net.forward(['detection_out_final','detection_masks'])

    boxes = boxes[0,0,:,:]

    boxesToDraw = []
    h,w = img.shape[:2]
    for box,mask in zip(boxes,masks) :
        _,classid,score,x1,y1,x2,y2 = box
        classid = int(classid)
        color = colors[classid]
        if score > 0.6 :
            x1 = max(0, min(int(x1 * w),w-1)); y1 = max(0, min(int(y1 * h),h-1))
            x2 = max(0, min(int(x2 * w),w-1)); y2= max(0, min(int(y2 * h),h-1))

            boxesToDraw.append([img,classid,score,x1,y1,x2,y2])
            classMask = mask[classid]

            classMask = cv2.resize(classMask, (x2-x1+1,y2-y1+1))
            mask = (classMask > 0.3)
        
            roi = img[y1:y2+1,x1:x2+1][mask]

            img[y1:y2+1,x1:x2+1][mask] = (0.7 * color + 0.3 * roi).astype(np.uint8)
            
            # 이렇게 그려줘도 무방.
            # label = f"{classNames[classid]} : {score:.2f}"
            # cv2.rectangle(img, (x1,y1),(x2,y2), color,2,cv2.LINE_AA)
            # cv2.putText(img, label,(x1,y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,color,1,cv2.LINE_AA)


    for box in boxesToDraw :
        drawBox(*box)

    t, _ = net.getPerfProfile()
    label = f"Inference time : {t*1000.0 / cv2.getTickFrequency():.2f} ms"
    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),1,cv2.LINE_AA)


    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()




