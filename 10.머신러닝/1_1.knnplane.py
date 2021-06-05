import numpy as np
import cv2


def addPoint(x,y,c) :
    train.append([x,y])
    label.append([c])

def on_k_changed(pos) :
    global k_value 
    k_value = pos
    if k_value < 1 :
        k_value = 1

    trainAndDisplay()

def trainAndDisplay() :
    trainData = np.array(train,np.float32)
    labelData = np.array(label,np.int32)

    knn.train(trainData,cv2.ml.ROW_SAMPLE,labelData)

    h,w = img.shape[:2]
 
    # 예시를 위한 코드임. -> 좋은 코드는 아니다. 
    for y in range(h) :
        for x in range(w) :
            sample = np.array([[x,y]]).astype(np.float32)

            # _,ret,_,_ = knn.findNearest(sample,k_value)
            ret,_,_,_ = knn.findNearest(sample,k_value)
            
            # ret = int(ret[0,0])
            ret = int(ret)

            if ret == 0 :
                img[y,x] = (128,128,255)
            elif ret == 1 :
                img[y,x] = (128,255,128)
            elif ret == 2 :
                img[y,x] = (255,128,128)
        
    for i in range(len(train)) :
        x,y = train[i]
        c = label[i][0]

        if c == 0 :
            cv2.circle(img, (x,y), 5, (0,0,128),-1,cv2.LINE_AA)
        elif c == 1 :
            cv2.circle(img, (x,y), 5, (0,128,0),-1,cv2.LINE_AA)
        elif c == 2 :
            cv2.circle(img, (x,y), 5, (128,0,0),-1,cv2.LINE_AA)

    cv2.imshow('knn', img)

# 학습과 라벨 데이터
train = []
label = []

k_value = 1
img = np.full((500,500,3), 255,np.uint8)
knn = cv2.ml.KNearest_create()

# 랜덤 데이터 생성
MAXN = 30
rn = np.zeros((MAXN,2),np.int32)

# 정규분포된 난수 구하는 함수
# (150,150) 근방에 0번 클래스
cv2.randn(rn,0,50)
for x,y in rn :
    addPoint(x+150,y+150,0)

# (350,150) 근방에 1번 클래스
cv2.randn(rn,0,50)
for x,y in rn :
    addPoint(x+350,y+150,1)

# (250,400) 근방에 2번 클래스
cv2.randn(rn,0,50)
for x,y in rn :
    addPoint(x+250,y+400,2)

# 영상 출력 창 생성 & 트랙바 생성
cv2.namedWindow('knn')
cv2.createTrackbar('k_value', 'knn', 1, 5, on_k_changed)

trainAndDisplay()

cv2.waitKey()
cv2.destroyAllWindows()