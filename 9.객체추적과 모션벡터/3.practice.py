import sys
import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened() :
    print("Video load falied")
    sys.exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w2 = w // 2
h2 = h // 2 

_ , frame = cap.read()

frame = cv2.flip(frame,1)
gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (w2,h2),interpolation=cv2.INTER_AREA)

while True :
    ret, frame = cap.read()
    if not ret : break

    frame = cv2.flip(frame,1)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.resize(gray2, (w2,h2),interpolation=cv2.INTER_AREA)

    # 옵티컬 플로우 계산
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 13, 3, 5, 1.1, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # 움직임 벡터 시각화
    hsv = np.zeros((h2,w2,3),dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # 움직임이 충분히 큰 영역 선택
    motion_mask = np.zeros((h2,w2),dtype=np.uint8)
    motion_mask[mag > 2.0] = 255

    mx = cv2.mean(flow[...,0],mask=motion_mask)[0]
    my = cv2.mean(flow[...,1],mask=motion_mask)[0]
    m_mag = np.sqrt(mx*mx + my*my)

    # 충분히 큰 영역
    if m_mag > 4.0 :
        # 이동방향의 각 구하기
        m_ang = np.arctan2(my,mx) * 180 / np.pi
        m_ang += 180 

        pt1 = (100,100)
        if 45 <= m_ang < 135 : 
            pt2 = (100,30)
        elif 135 <= m_ang < 225 :
            pt2 = (170,100)
        elif 255 <= m_ang < 315 :
            pt2 = (100,170)
        else :
            pt2 = (30,100)
        
        cv2.arrowedLine(frame, pt1, pt2, (0,0,255),7,cv2.LINE_AA,tipLength=0.7)
    
    cv2.imshow('frame', frame)
    # cv2.imshow('flow', bgr)

    if cv2.waitKey(1) == 27 :
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()