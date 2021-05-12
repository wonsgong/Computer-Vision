import sys
import numpy as np
import cv2

def drawROI(img,corners) :
    cpy = img.copy()
    c1 = (192, 192, 255)
    c2 = (128, 128, 255)
    # 모서리 점 좌표에 원 그리기
    for pt in corners :
        cv2.circle(cpy, tuple(pt), 25, c1,-1,cv2.LINE_AA)

    # 박스 그려주기
    cv2.line(cpy, tuple(corners[0]), tuple((corners[1])), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1]), tuple((corners[2])), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2]), tuple((corners[3])), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3]), tuple((corners[0])), c2, 2, cv2.LINE_AA)
    dst = cv2.addWeighted(img,0.3,cpy,0.7,0)

    return dst

def onMouse(event,x,y,flags,param):
    global srcQuad, dragFlag, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN :
        for i in range(4) :
            if cv2.norm(srcQuad[i] - (x,y)) < 25 :
                dragFlag[i] = True
                ptOld = (x,y)
                break
    
    if event == cv2.EVENT_LBUTTONUP :
        for i in range(4):
            dragFlag[i] = False
        
    if event == cv2.EVENT_MOUSEMOVE :
        for i in range(4) :
            if dragFlag[i] == True :
                dx = x - ptOld[0]
                dy = y - ptOld[1]
                srcQuad[i] += (dx,dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow('img',cpy)
                ptOld = (x,y)
                break

src = cv2.imread('image/scanned.jpg')

if src is None:
    print("Image load failed")
    sys.exit()


h,w = src.shape[:2]
dw = 500
dh = round(dw * 297 / 210) # A4 용지 크기 : 210x297

# 모서리 점들의 좌표.
srcQuad = np.array([(30, 30), (30, h-30),(w-30, h-30),(w-30, 30)], dtype=np.float32)
dstQuad = np.array([(0, 0), (0, dh-1), (dw-1, dh-1),(dw-1, 0)], dtype=np.float32)
dragFlag = [False,False,False,False]


dst = drawROI(src, srcQuad)
cv2.imshow('img', dst)
cv2.setMouseCallback('img', onMouse)

while True :

    key = cv2.waitKey()

    if key == 13 : # Enter
        break
    if key == 27 : # ESC
        cv2.destroyAllWindows()
        sys.exit()
    
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
dst = cv2.warpPerspective(src, pers, (dw,dh))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
