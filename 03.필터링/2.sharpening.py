import sys
import numpy as np
import cv2

# 그레이스케일 영상
src = cv2.imread("image/rose.bmp",cv2.IMREAD_GRAYSCALE)

if src is None :
    print("Image load failed")
    sys.exit()


blr = cv2.GaussianBlur(src,(0,0),2)
dst = cv2.addWeighted(src,2,blr,-1,0)
# dst = np.clip(2.0 * src - blr,0,255).astype(np.uint8)

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey()
cv2.destroyAllWindows()

# 컬러 영상
src = cv2.imread("image/rose.bmp")

if src is None:
    print("Image load failed")
    sys.exit()

src_ycrcb = cv2.cvtColor(src,cv2.COLOR_BGR2YCrCb)

# 연산 시 보다 정밀하게 하기 위해 실수 변환.
# 연살 할 땐 실수 , 결과 출력 시엔 정수로 하는게 좋다.
src_y = src_ycrcb[:,:,0].astype(np.float32)
blr = cv2.GaussianBlur(src_y,(0,0),2)
src_ycrcb[:,:,0] = np.clip(2.0 * src_y - blr,0,255).astype(np.uint8)

dst = cv2.cvtColor(src_ycrcb,cv2.COLOR_YCrCb2BGR)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
