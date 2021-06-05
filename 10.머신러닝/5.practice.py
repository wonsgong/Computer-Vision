import numpy as np
import cv2
import sys

def normImg(img) :
    h,w = img.shape[:2]

    # img : src to gray 이미지. 반전 필요.
    img = ~img

    # scale factor, 20 * 20 이기때문에 20이여야 하지만 여백(3픽셀씩) 고려
    sf = 14. / h
    if w > h : sf = 14. / w

    img2 = cv2.resize(img, (0,0),fx=sf,fy=sf,interpolation=cv2.INTER_AREA)

    h2, w2 = img2.shape[:2]
    a = (20 - w2) // 2
    b = (20 - h2) // 2

    # 20 by 20 이미지에 중앙에 img2 복사
    dst = np.zeros((20,20),np.uint8)
    dst[b:b+h2,a:a+w2] = img2[:,:]

    return dst


src = cv2.imread('image/handwritten1.png')

if src is None :
    print("Image load failed")
    sys.exit()

# HOG 생성 & svm 읽어오기.
hog = cv2.HOGDescriptor((20,20),(10,10),(5,5),(5,5),9)
svm = cv2.ml.SVM_load('svmdigit.yml')

# 이진화 & 레이블링
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_ , gray_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
_, _, stats, _ = cv2.connectedComponentsWithStats(gray_bin)

dst = src.copy()

for stat in stats[1:] :
    x,y,w,h,s = stat

    if s < 100 : continue

    testImg = normImg(gray[y:y+h,x:x+w])
    testImg = hog.compute(testImg).T

    _,ret = svm.predict(testImg)

    cv2.rectangle(dst, (x,y,w,h), (0,0,255),1,cv2.LINE_AA)
    cv2.putText(dst, str(int(ret[0,0])) ,(x-5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),4,cv2.LINE_AA)


cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

