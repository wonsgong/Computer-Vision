# OpenCV 기초 사용법
## 1. 영상의 속성과 픽셀 값 처리
opencv 에서는 영상 데이터를 numpy.ndarray 로 표현
```python 
	import cv2
	img1 = cv2.imread('cat.bmp', cv2.IMREAD_GARYSCALE)
	img2 = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)
```
`img1` 과 `img2` 는`np.ndarray` 형태. 
```python
	img1.shape # (h,w) -> 그레이스케일 영상
	img2.shape # (h,w,3) -> 컬러 영상
	# dtype 은 np.unit8 -> opencv자료형 : cv2.CV_8UC1(그레이) / CV_8UC3(컬러)
	# 영상 데이터 참조 방법
	h,w = img2.shape[:2] 
```

## 2. 영상의 생성, 복사, 부분 영상 추출
### 2-1. 지정한 크기로 새 영상 생성하기
``` python
	import numpy as np
	np.empty(shape, dtype, ...) -> arr # 임의의 값으로 초기화된 배열
	np.zeros(shape, dtype, ...) -> arr # 0 로 초기화된 배열
	np.ones(shape, dtype, ... ) -> arr # 1 로 초기화된 배열
	np.full(shape,fill_value, dtype, ...) -> arr # fill_value 로 초기화된 배열
	# shape : 각 차원의 크기.
	# dtype : 일반적인 경우 np.unit8 지정
	# arr : np.ndarray
```
### 복사
``` python
	img1 = cv2.imread('HappyFish.jpg')

	img2 = img1 # 1번  
	img3 = img1.copy() # 2번 
```
1번과 2번의 차이는 img2 변경 시 img1 도 변경됨. img3  변경 시 img1 변경 X
-> 메모리 자체를 넘겨주느냐 아니냐의 차이.
-> 안전한 방법을 사용해야 할때는 .copy() 를 사용.

### 부분 영상 추출
```python
	img1 = cv2.imread('HappyFish.jpg')

	img2 = img1[40:120,30:150] # 1번  
	img3 = img1[40:120,30:150.copy() # 2번 
```
부분 영상 추출은 `np.ndarray` 의 슬라이싱을 이용해서 작업.
1번과 같은 방법은 `ROI(Region Of Interest,관심영역)` 을 처리하는 데 이용하기도 한다.(mask 연산 시에 진행할 예정)

## 3. 마스크 연산과 ROI
### 3-1. ROI
* Region of Interest , 관심 영역
* 영상에서 특정 연산을 수행하고자 하는 임의의 부분 영역

### 3-2 마스크 연산
* OpenCV는 일부 함수에 대해서 ROI 연산을 지원하며, *마스크 영상* 을 인자로 전달해야함(`cv2.copyTo()` ,`cv2.calcHist()` ,`cv2.bitwise_or()` ,  등등)
* 마스크 영상은 `cv2.cv2_8UC1`(그레이 스케일 영상)
* 픽셀 값이 0이 아닌 위치에서만 연산이 수행됨. -> 0 or  255 로 구성된 이진 영상을 사용.
```python
cv2.copyTo(src,mask,dst=None) -> dst 
# 입력에 dst 를 주어야 src 에 mask 를 적용한 값을 dst 에 복사해준다.
```
* 특정 영역에 복사를 원한다면 부분 영상(1번 방식으로)을 추출해 ```dst``` 로 이용한다. 


## 4. 그리기 함수
* OpenCV에선 영상에 선, 도형, 문자열을 출력하는 그리기 함수를 제공
	* 선 그리기 : 직선, 화살표, 마커 등
	* 도형 그리기 : 사각형, 원, 타원, 다각형 등
	* 문자열 출력
* 주의할 점
	* 그리기 함수는 영상의 픽셀 값 자체를 변경한다 -> 원본 영상이 필요하면 복사본 만들어서 진행해야 한다.
	* 그레이스케일 영상에서는 컬러로 그리기 불가 -> `cv2.cvtColor()`로 컬러 변환 후 그리기

### 4-1 그리기 함수
* 직선
```python
cv2.line(img, pt1, pt2, color, thickness=None, lineType=None,shift=None) -> img
```
`img` : 직선 그릴 도화지
`pt1`,`pt2` : 직선의 시작점 과 끝점(x,y)
`color` : 선 색상 또는 밝기 . (B,R,G) or int
`thickness` : 선 두께. 기본값은 1
`lineType` : `cv2.LINE_4` , `cv2.LINE_8` , `cv2.LINE_AA` AA가 가장 부드러움.
`shift` : 그리기 좌표 값의 축소 비율, 기본 값 0 -> 특수한 경우에 사용.

* 사각형
```python
cv2.rectangle(img,pt1,pt2,color,thickness=None,lineType=None,shift=None) -> img
cv2.rectangle(img,rec,color,thickness=None,lineType=None,shift=None) -> img
```
`rec` : 사각형 위치 정보. (x,y,w,h) 
`thickness` : -1 지정 시 내부를 채운다.

* 원
```python
cv2.circle(img, center, radius, color, thickness=None, lineType=None,shift=None) -> img
```
`center` : 중심 좌표, (x,y)
`radius` : 반지름
`lineType` : 원과 같은 곡선은 `cv2.LINE_AA` 로 그리는 게 가장 부드럽다.

* 다각형 그리기
```python
cv2.polylines(img, pts, isClosed, color, thickness=None, lineType=None,shift=None) -> img
```
`pts` : 외곽 점들의 좌표 배열. `np.ndarray` 의 리스트.
`isClosed` : 폐곡선 여부. T/F

* 텍스트 
```python
cv2.putText(img, text, org, fontFace, fontScale, thickness=None, lineType=None, bottomLeftOrigin=None) -> img
```
`text` : 출력할 문자열
`org` : 문자열 출력할 위치의 좌측 하단 좌표
`fontFace` : 폰트 종류. `cv2.FONT_HERSHEY_` 로 시작하는 상수. 한글은 지원안함.
`fontScale` : 폰트 크기
`bottomLeftOrigin` : 좌측 하단을 원점으로 간주, T/F -> 기본값은 True
