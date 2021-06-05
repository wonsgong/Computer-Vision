import sys
import numpy as np
import cv2
import pytesseract


def reorderPts(pts) :
    # col 0, col1 기준 정렬. 뒤가 우선순위가 높다
    idx = np.lexsort((pts[:,1],pts[:,0]))
    pts = pts[idx]

    if pts[0,1] > pts[1,1] :
        pts[[0,1]] = pts[[1,0]]
    if pts[2,1] < pts[3,1] :
        pts[[2,3]] = pts[[3,2]]
    return pts

img = cv2.imread('image/namecard2.jpg')

if img is None :
    print("Image load failed")
    sys.exit()

# 출력 영상 설정
dw,dh = 720,400
srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]],np.float32)
dstQuad = np.array([[0, 0], [0, dh], [dw, dh], [dw, 0]],np.float32)


# 영상 이진화
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

# 외곽선 검출
contours , _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 사각형 검출
cpy = img.copy()
for pts in contours :
    
    # 작은 면적은 날려버려
    if cv2.contourArea(pts) < 400 : continue

    # 다각형 근사화
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts,True)*0.02, True)

    # 사각형인 경우
    if len(approx)== 4 :
        cv2.polylines(img, [approx], True, (0,0,255),2,cv2.LINE_AA)
        srcQuad = reorderPts(approx.reshape(4,2).astype(np.float32))
        break


pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
dst = cv2.warpPerspective(img, pers, (dw,dh))

# 글씨 검출
dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
print(pytesseract.image_to_string(dst_gray,lang='kor+eng'))

cv2.imshow('img',img)
cv2.imshow('dst',dst)

cv2.waitKey()
cv2.destroyAllWindows()


