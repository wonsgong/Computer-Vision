import numpy as np
import cv2
import sys

# 실습을 위한 데이터
trains = np.array([[150, 200], [200, 250],
                   [100, 250], [150, 300],
                   [350, 100], [400, 200],
                   [400, 300], [350, 400]], dtype=np.float32)
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setKernel(cv2.ml.SVM_RBF)

svm.trainAuto(trains,cv2.ml.ROW_SAMPLE,labels)
print("C : ",svm.getC())
print("Gamma : ",svm.getGamma())

# 시각화를 위한 부분
w, h = 500, 500
img = np.zeros((h,w,3),np.uint8)

for y in range(h) :
    for x in range(w) :
        test = np.array([[x,y]],np.float32)
        _, ret = svm.predict(test)

        ret = int(ret[0,0])

        if ret == 0 :
            img[y,x] = (128,128,255)
        else :
            img[y,x] = (128,255,128)
    

color = [(0,0,128),(0,128,0)]
for i in range(len(trains)) :
    x = int(trains[i,0])
    y = int(trains[i,1])
    c = labels[i]

    cv2.circle(img, (x,y), 4, color[c],-1,cv2.LINE_AA)


cv2.imshow('img', img)
cv2.waitKey()

cv2.destroyAllWindows()

