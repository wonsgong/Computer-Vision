import numpy as np
import cv2
import sys

# 무게 중심 구해서 이동 변환.
def normImg(img) :
    m = cv2.moments(img)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    h,w = img.shape[:2]
    aff = np.array([[1,0,w/2 - cx],[0,1,h/2 - cy]],np.float32)
    dst = cv2.warpAffine(img, aff, (0,0))

    return dst

oldx , oldy = -1,-1
def onMouse(event,x,y,flags,_) :
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN :
        oldx,oldy = x,y
    
    if event == cv2.EVENT_LBUTTONUP : 
        oldx,oldy = -1,-1
    
    if event == cv2.EVENT_MOUSEMOVE :
        if flags & cv2.EVENT_FLAG_LBUTTON :
            cv2.line(img,(oldx,oldy),(x,y),255,40,cv2.LINE_AA)
            oldx,oldy = x,y
            cv2.imshow('img',img)



digits = cv2.imread("image/digits.png",cv2.IMREAD_GRAYSCALE)

if digits is None :
    print("Image load failed")
    sys.exit()

h,w = digits.shape[:2]
hog = cv2.HOGDescriptor((20,20),(10,10),(5,5),(5,5),9)

cells = np.array([np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)])
cells = cells.reshape(-1,20,20)

desc = []
# 이미지 위치 정규화
for cell in cells :
    cell = normImg(cell)
    desc.append(hog.compute(cell))

desc_train = np.array(desc).squeeze().astype(np.float32)
desc_label = np.repeat(np.arange(10), len(desc_train)/10)
print(desc_train.shape)

# 이하 내용은 2_2.svmdigit 과 같음
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)

svm.setC(2.5)
svm.setGamma(0.50625)
svm.train(desc_train,cv2.ml.ROW_SAMPLE,desc_label)

svm.save('svmdigit.yml')

img = np.zeros((500,500),np.uint8)
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)

while True :
    key = cv2.waitKey()

    if key == 27 :
        break

    elif key == ord(' ') :
        testImg = cv2.resize(img, (20,20),interpolation=cv2.INTER_AREA)
        testImg = normImg(testImg)
        testImg = hog.compute(testImg).T

        _, ret = svm.predict(testImg)
        print(int(ret[0,0]))

        img.fill(0)
        cv2.imshow('img', img)


cv2.destroyAllWindows()









