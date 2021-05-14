import cv2
import numpy as np
import sys

src = cv2.imread('image/rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

dh = src.shape[0] // 4
dw = src.shape[1] // 4

# 지역 이진화 진행(local_th.py 참고)
dst1 = np.zeros(src.shape, np.uint8)
for y in range(4):
    for x in range(4):
        src_ = src[y*dh:(y+1)*dh, x*dw:(x+1)*dw]
        dst_ = dst1[y*dh:(y+1)*dh, x*dw:(x+1)*dw]
        cv2.threshold(src_, 0, 255, cv2.THRESH_OTSU, dst_)

# 객체(흰색영역) 카운트 해주는 함수
cnt1, _ = cv2.connectedComponents(dst1)
print("cnt1 : ",cnt1)

# 노이즈가 제거됨. 
# dst2 = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, None)
dst2 = cv2.erode(dst1, None)
dst2 = cv2.dilate(dst2, None)
cnt2, _ = cv2.connectedComponents(dst2)
print("cnt2 : ",cnt2)


cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()


cv2.connectedComponentsWithStats(image)