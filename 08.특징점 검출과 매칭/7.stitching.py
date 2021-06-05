import sys
import cv2
import numpy as np

img_names = ['image/img1.jpg','image/img2.jpg','image/img3.jpg']
imgs = []
for name in img_names :
    img = cv2.imread(name)
    
    if img is None :
        print("Image load failed")
        sys.exit()
    

    imgs.append(img)


Stitcher = cv2.Stitcher_create()
ret, pano = Stitcher.stitch(imgs)

if ret == cv2.STITCHER_OK :
    cv2.imshow('pano', pano)   
    cv2.waitKey()
    cv2.destroyAllWindows()
else :
    print("Stitch fail")
    sys.exit()
