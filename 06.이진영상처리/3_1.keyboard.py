import cv2
import numpy as np
import sys

src = cv2.imread('image/keyboard.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

cnt, labels, stats, centorids = cv2.connectedComponentsWithStats(src_bin)

# 0은 배경이니까 1부터 
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
for i in range(1,cnt) :
    x,y,w,h,a = stats[i]

    # 면적 크기를 활용해서 노이즈 제거
    if a < 20 : continue 
    cv2.rectangle(dst, (x,y,w,h), (0,0,255),2)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()