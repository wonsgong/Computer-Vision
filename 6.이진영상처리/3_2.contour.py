import sys
import numpy as np
import cv2
import random

src = cv2.imread('image/polygon.bmp')

if src is None:
    print("Image load failed")
    sys.exit()

# 계층 구조 사용.
contour, hier = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

dst1 = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
idx = 0
# next 를 이용해서 그리기
while idx >= 0 :
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst1, contour,idx, c,2,cv2.LINE_AA,hier)
    # next 의미
    idx = hier[0,idx,0]


# 계층구조 없이 LIST 로
contour, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
# contour 를 전부 돌면서 그리기.
for idx in range(len(contour)) :
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst2, contour, idx, c, 2, cv2.LINE_AA)

cv2.imshow('CCOMP', dst1)
cv2.imshow('LIST', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
