import sys
import numpy as np
import cv2

src = cv2.imread("image/nemo.jpg")

if src is None :
    print("Image load failed")
    sys.exit()

# Use cv2.GC_INIT_WITH_RECT
# select ROI
rc = cv2.selectROI(src)
mask = np.zeros(src.shape[:2],np.uint8)

cv2.grabCut(src, mask, rc, None, None, 5,cv2.GC_INIT_WITH_RECT)

# cv2.GC_BGD(0) , cv2.GC_PR_BGD(2)
mask2 = np.where((mask == 0) | (mask == 2),0,1).astype(np.uint8)
dst = src * mask2[:,:,np.newaxis]

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

# Use cv2.GC_INIT_WITH_MASK (마우스 없어서 마우스 있는 환경에서 진행)
