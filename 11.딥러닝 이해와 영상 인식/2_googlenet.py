import sys
import numpy as np
import cv2

root = "image/{}"
imgName = ["beagle.jpg","cup.jpg","pineapple.jpg","scooter.jpg","space_shuttle.jpg"]

imgs = []
for n in imgName :
    img = cv2.imread(root.format(n))

    if img is None :
        print("Image load failed")
        sys.exit()

    imgs.append(img)


# model = "dnn/bvlc_googlenet.caffemodel"
# config = "dnn/deploy.prototxt"

model = "dnn/googlenet-9.onnx"
config = ""

modelName = None 
with open("dnn/classification_classes_ILSVRC2012.txt") as f :
    modelName = f.read().rstrip("\n").split("\n")

net = cv2.dnn.readNet(model,config)
for img in imgs :
    blob = cv2.dnn.blobFromImage(img,1,(224,224),(104,117,123))
    net.setInput(blob)
    out = net.forward()
    idx = np.argmax(out)

    name = modelName[idx]
    prob = round(out[0][idx] * 100,2)

    cv2.putText(img, str(name)+" "+str(prob)+"%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1,cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()
cv2.destroyAllWindows()

