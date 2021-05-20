import sys
import numpy as np
import cv2

src = cv2.imread("image/circuit.bmp",cv2.IMREAD_GRAYSCALE)
templ = cv2.imread("image/crystal.bmp",cv2.IMREAD_GRAYSCALE)

if src is None or templ is None :
    print("Image load failed")
    sys.exit()

# 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가
noise = np.zeros(src.shape,np.int32)
cv2.randn(noise,50,10)
src = cv2.add(src,noise,dtype=cv2.CV_8UC3)

res = cv2.matchTemplate(src, templ, cv2.TM_CCOEFF_NORMED)
res_norm = cv2.normalize(res, None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

_, maxv, _, maxloc = cv2.minMaxLoc(res)
print("maxv:",maxv)
print("maxloc",maxloc)

# 매칭 결과 표시
th, tw = templ.shape[:2]
dst = cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)
cv2.rectangle(dst, maxloc, (maxloc[0]+tw,maxloc[1]+th), (0,0,255),2)

cv2.imshow('dst',dst)
cv2.imshow('res', res_norm)
cv2.waitKey()
cv2.destroyAllWindows()