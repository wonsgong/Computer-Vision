import sys
import numpy as np
import cv2

src = cv2.imread("image/lenna.bmp")

if src is None :
    print("Image load failed")
    sys.exit()


classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

if classifier.empty() :
    print("XML load failed")
    sys.exit()


res = classifier.detectMultiScale(src)

for rc in res :
    cv2.rectangle(src, rc, (0,0,255),2)

cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()