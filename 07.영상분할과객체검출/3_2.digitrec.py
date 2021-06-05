import sys
import numpy as np
import cv2


# load digit
src = cv2.imread("image/digits_print.bmp",cv2.IMREAD_GRAYSCALE)
digits = []
for i in range(10) :
    root = "image/digits/digit{}.bmp"
    digits.append(cv2.imread(root.format(i),cv2.IMREAD_GRAYSCALE))

    if digits[i] is None :
        print("Image load failed")
        sys.exit()

if src is None  :
    print("Image load failed")
    sys.exit()


_,src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
src_contours,_ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dst = cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)
for pts in src_contours :
    if cv2.contourArea(pts) < 1000 : continue

    x,y,w,h = cv2.boundingRect(pts)
    
    crap = src[y:y+h, x:x+w]

    crap = cv2.resize(crap,(100,150))
    res = []
    for digit in digits :
        res.append(cv2.matchTemplate(crap, digit, cv2.TM_CCOEFF_NORMED))

    n = np.argmax(res)
    cv2.putText(dst, str(n), (x,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2,cv2.LINE_AA)
    cv2.rectangle(dst,(x,y,w,h),(255,0,0),1)


cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()