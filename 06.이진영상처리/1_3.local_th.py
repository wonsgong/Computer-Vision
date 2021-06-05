import cv2
import numpy as np
import sys

src = cv2.imread('image/sudoku.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

# shape 을 고려해서 적절하게 나눠줘야 한다. -> 전부다 커버가 안될경우 남은 부분을 따로 해줘야한다.
dh = src.shape[0] // 4
dw = src.shape[1] // 4

_,dst1 = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

dst2 = np.zeros(src.shape,np.uint8)

for y in range(4) :
    for x in range(4) :
        src_ = src[y*dh:(y+1)*dh,x*dw:(x+1)*dw]
        dst_ = dst2[y*dh:(y+1)*dh,x*dw:(x+1)*dw]
        cv2.threshold(src_, 0, 255, cv2.THRESH_OTSU,dst_)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()


