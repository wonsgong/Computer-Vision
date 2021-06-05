import sys
import numpy as np
import cv2

# 히스토그램 스트레칭


src = cv2.imread('image/lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

def getGrayHistImage(hist) :
    imgHist = np.full((100,256),255,dtype=np.uint8)
    
    histMax = np.max(hist) 
    for x in range(256) :
        pt1 = (x,100)
        pt2 = (x,100 - int(hist[x,0] * 100 / histMax))
        cv2.line(imgHist,pt1,pt2,0)

    return imgHist

# 기존 이미지 히스토그램
srchist = cv2.calcHist([src], [0], None, [256], [0,256])
srchistimg = getGrayHistImage(srchist)

# 1번째 방식 dst(x,y) = saturate((1+alpha)src - 128alpha))
alpha = 1.0
dst1 = np.clip((1+alpha)*src - 128*alpha, 0, 255).astype(np.uint8)

# 2번째 방식 normalize
dst2 = cv2.normalize(src, None,0,255,cv2.NORM_MINMAX)
dst2hist = cv2.calcHist([dst2], [0], None, [256], [0,256])
dst2histimg = getGrayHistImage(dst2hist)

# 3번쨰 방식 직선의 방정식
gmin = np.min(src)
gmax = np.max(src)
dst3 = np.clip(((src-gmin) * 255.) / (gmax-gmin),0,255).astype(np.uint8)
dst3hist = cv2.calcHist([dst3], [0], None, [256], [0,256])
dst3histimg = getGrayHistImage(dst3hist)



cv2.imshow('src', src)
cv2.imshow('srchist', srchistimg)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst2hist',dst2histimg)
cv2.imshow('dst3', dst3)
cv2.imshow('dst3hist',dst3histimg)
cv2.waitKey()

cv2.destroyAllWindows()
