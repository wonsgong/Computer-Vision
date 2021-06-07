import sys
import numpy as np
import cv2

oldx, oldy = -1,-1

def onMouse(event, x, y, flags, _) :
    global oldx,oldy
    if event == cv2.EVENT_LBUTTONDOWN :
        oldx,oldy = x,y
    
    if event == cv2.EVENT_LBUTTONUP :
        oldx,oldy = -1,-1

    if event == cv2.EVENT_MOUSEMOVE :
        if flags & cv2.EVENT_FLAG_LBUTTON :
            cv2.line(img, (oldx,oldy), (x,y), 255,20,cv2.LINE_AA)
            oldx,oldy = x,y
            cv2.imshow('img', img)

def norm_hangul(img):
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    dst = cv2.warpAffine(img, aff, (0, 0))
    return dst

net = cv2.dnn.readNet("dnn/korean_recognition.pb")

if net.empty() :
    print("Err")
    sys.exit()

 
hangulName = None 
with open("dnn/256-common-hangul.txt","rt",encoding='utf-8') as f :
    hangulName = f.read().rstrip("\n").split("\n")

img = np.zeros((500,500),np.uint8)
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)

while True :
    key = cv2.waitKey()

    if key == 27 : 
        break

    elif key == ord(' ') :
        blob = cv2.dnn.blobFromImage(img,1,(64,64))
        net.setInput(blob)
        out = net.forward()

        idx = np.argmax(out)
        name = hangulName[idx]
        prob = out[0][idx]

        print("Character is {} , {}%".format(name,round(prob*100,2)))

        img.fill(0)

        cv2.imshow('img', img)

cv2.destroyAllWindows()

