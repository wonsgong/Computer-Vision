import numpy as np
import cv2
import sys

# 픽셀값 자체를 주는게 좋은 것은 아니다.

oldx , oldy = -1,-1
def onMouse(event, x, y, flags, _) :
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN :
        oldx , oldy = x , y 

    if event == cv2.EVENT_LBUTTONUP :
        oldx , oldy = -1, -1

    if event == cv2.EVENT_MOUSEMOVE :
        if flags & cv2.EVENT_FLAG_LBUTTON :
            cv2.line(img, (oldx,oldy), (x,y), (255,255,255), 40, cv2.LINE_AA)
            oldx , oldy = x , y 
            cv2.imshow('img', img)


digits = cv2.imread("image/digits.png",cv2.IMREAD_GRAYSCALE)

if digits is None :
    print("Image load failed")
    sys.exit()

h,w = digits.shape[:2]
# 학습 데이터 & 라벨링 
cells = [np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)]
cells = np.array(cells)

trainImg = cells.reshape(-1,400).astype(np.float32)
trainLabel = np.repeat(np.arange(10), len(trainImg)/10)

# Knn 학습
knn = cv2.ml.KNearest_create()
knn.train(trainImg,cv2.ml.ROW_SAMPLE,trainLabel)

# 사용자 입력 이미지
img = np.zeros((400,400),np.uint8)
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)

while True :

    key = cv2.waitKey()

    if key == 27 :
        break

    if key == ord(' ') :
        testImg = cv2.resize(img,(20,20),interpolation=cv2.INTER_AREA)
        testImg = testImg.reshape(-1,400).astype(np.float32)
        ret,_,_,_ = knn.findNearest(testImg,5)

        img.fill(0)
        print(int(ret))
        cv2.imshow('img', img)
       

cv2.destroyAllWindows()
