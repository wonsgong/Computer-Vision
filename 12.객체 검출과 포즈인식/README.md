# 12. 딥러닝 활용. 객체 검출, 포즈 인식

## 1. OpenCV DNN 얼굴 검출
>SSD(Single Shot MultiBox Detector) 기반 얼굴 검출 네트워크 [링크](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)  
> 기존의 `Haar-Cascade` 방법보다 속도 & 정확도 면에서 더 좋은 성능을 보여준다.  

* SSD(Single Shot MultiBox Detector) (W. Liu, et. al, 2016)
	동시대 다른 객체 검출 알고리즘과 비교하여 성능과 속도 두 가지를 모두 만족시킴. [관련 논문](https://arxiv.org/pdf/1512.02325.pdf).  
	![ssd](./image/ssd.png).  
	입력 :  
	`size=(300,300)` , `scale= 1[0,255]` , `Mean=(104,177,123)` , `RGB=False`
	출력 :  
	`out.shape=(1,1,N,7)` -> 뒤 두개만 필요`detect=out[0,0,:,:]`.  
	`detect[0,:] = [0,1,c,x1,y1,x2,y2]`  
	`c = 확률` , `x1,y1 = 좌상단` `x2,y2 = 우하단` . 좌표는 [0,1]로 정규화되어있기때문에 복원(frame 크기만큼 곱해줘)해서 써야된다.  
	 
## 2. YOLO 객체 검출
> You Only Look Once , 실시간 객체 검출 알고리즘. [사이트](https://pjreddie.com/darknet/yolo)

* YOLOv3 
	2018년 4월에 발표된 [Tech Report](https://pjreddie.com/darknet/yolo/)  
	기존 객체 검출 방법과 성능은 비슷하고 **속도는 훨씬 빠르다.**  
	[COCO 데이터셋](https://cocodataset.org) 사용(80개 클래스 객체 검출)  
	![YOLO](./image/YOLO.png).  
	입력 :  
	`size` : `(320,320)` / `(416,416)` / `(608,608)` , 속도-정확도 trade-off.  
	`scale= 1/255.` , `Mean=(0,0,0)` , `RGB=True`.  
	출력 :  
	3개의 출력 레이어.  
	`outs[0].shape=(507,85)` / `outs[1].shape=(2028,85)` / `outs[1].shape=(8112,85)`.  
	`outs[0][0,:] = [tx,ty,tw,th,p,p1...p80]`.  
	`tx,ty,tw,th` : 바운딩 박스의 중앙, 크기. `p` : 확률.  `p1-p80` : 클래스 스코어.  
	 

* 출력 레이어 받아오기.
	```python
	# 출력 레이어 이름 받아오기
	layerName = net.getLayerNames()
	outLayer = [layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]
	#outLayer = ['yolo_82', 'yolo_94', 'yolo_106']
	#(중략)
	outs = net.forward(outLayer)
	# outs 는 82,94,106번째 레이어를 같게 된다.
	```
* 비최대 억제(NMS, Non-Maximum Suppression)
	바운딩 박스의 겹치는 정도를 가지고 지역을 설정. 이후 score_thre 를 통해 값을 확인.  
	`cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold) -> retval`.  
	`bboxes` : 바운딩 박스 정보 리스트.  
	`scores` : 스코어 리스트.  
	`score_thre` : 스코어 >= thre.  
	`nms_thre` : 바운딩 박스 겹치는 정도.  -> 지역 선정.  
	`retval` : 선별 인덱스. `shape=(N,1)`.  

* 연산 시간 측정
	`cv2.dnn_Net.getPerfProfile() -> retval, timings`.  
	`retval` : 시간.
	`timings` : Layer 별 시간.
	```python
	# 사용 예제
	t, _ = net.getPerfProfile()
	times = (t * 1000.0 / cv2.getTickFrequency()) 
	```
> outs 에 대한 후처리를 잘해줘야 한다.  

## 3. Mask-RCNN 영역 분할

* 영역 분할
	객체의 바운딩 박스 + 픽셀 단위 클래스 분류까지. -> 객체 윤곽 구분.  
	Semantic segmentation : 하나의 클래스는 모두 같은 레이블  
	Instance segmentation : 객체 단위 다른 레이블 [참고](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)  
	![seg](./image/seg.png).  

* Mask-RCNN
	대표적인 객체 영역 분할 딥러닝 알고리즘 (He et. al. 2017).  
	Faster R-CNN (object detection) + FCN (semantic segmentation).  
	[자세한 내용은 논문 참고](https://arxiv.org/pdf/1703.06870.pdf)  

	입력 :  
	`size = 임의(auto resize)`, `scale= 1` , `Mean=(0,0,0)` , `RGB=True`.  
	출력 :  
	2개의 출력 레이어.  
	`detection_out_final` -> `shape=(1,1,100,7)`.  
	=> `[0,classid,conf,x1,y1,x2,y2]`.  
	`detection_masks` -> `shape=(100,90,15,15)`.  
	=> 15 x 15 크기의 마스크.  

> 마스크 영역을 표시해줄 때, 실제 이미지 크기로 마스크를 리사이즈 후 표시해주어야 한다.!!!



