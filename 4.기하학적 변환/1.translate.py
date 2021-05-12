import sys
import numpy as np
import cv2



src = cv2.imread('image/tekapo.bmp')

if src is None :
    print("Image load failed")
    sys.exit()


trans = np.array([[1,0,200],
                  [0,1,100]],dtype=np.float32)

shear = np.array([[1,0.5,0],
                  [0,1,0]],dtype=np.float32)

dst = cv2.warpAffine(src, trans, (0,0))
dst2 = cv2.warpAffine(src, shear,(0,0))

# 잘리지 않게 다 출력하기 위해선 크기를 적절히 조절해줘야함.
h,w = src.shape[:2]
dst3 = cv2.warpAffine(src, trans,(w+200,h+100))
dst4 = cv2.warpAffine(src, shear,(w+int(h*0.5),h))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2',dst2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)
cv2.waitKey()
cv2.destroyAllWindows()
