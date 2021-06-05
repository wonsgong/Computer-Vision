import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# matplot 없이 히스토그램 출력
def getGrayHistImage(hist) :
    imgHist = np.full((100,256),255,dtype=np.uint8)
    
    histMax = np.max(hist) 
    for x in range(256) :
        pt1 = (x,100)
        pt2 = (x,100 - int(hist[x,0] * 100 / histMax))
        cv2.line(imgHist,pt1,pt2,0)

    return imgHist


# 그레이스케일 영상의 히스토그램
src = cv2.imread('image/lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

hist = cv2.calcHist([src], [0], None, [256], [0, 256])
imgHist = getGrayHistImage(hist)
cv2.imshow('src', src)
cv2.imshow('imgHist',imgHist)
cv2.waitKey(1)

plt.plot(hist)
plt.show()


# 컬러 영상의 히스토그램
src = cv2.imread('lenna.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

colors = ['b', 'g', 'r']
bgr_planes = cv2.split(src)

for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(hist, color=c)

cv2.imshow('src', src)
cv2.waitKey(1)

plt.show()

cv2.destroyAllWindows()
