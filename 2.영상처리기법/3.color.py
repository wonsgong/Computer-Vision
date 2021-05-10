import sys
import cv2
import matplotlib.pyplot as plt
# 컬러 영상 불러오기
src = cv2.imread('image/candies.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    sys.exit()

# 컬러 영상 속성 확인
print('src.shape:', src.shape)  # src.shape: (480, 640, 3)
print('src.dtype:', src.dtype)  # src.dtype: uint8


# RGB 색 평면 분할
b, g, r = cv2.split(src)
# b = src[:, :, 0]
# g = src[:, :, 1]
# r = src[:, :, 2]

# BGR to HSV
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(src_hsv)

# BGR to YCrCb
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y,cr,cb = cv2.split(src_ycrcb)

# plt 쓰기 위해 rbg 로 변환
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
plt.subplot2grid((3,4),(0,0)), plt.axis('off'), plt.imshow(src_rgb), plt.title('RGB')
plt.subplot2grid((3,4),(0,1)), plt.axis('off'), plt.imshow(b, 'gray'), plt.title('B')
plt.subplot2grid((3,4),(0,2)), plt.axis('off'), plt.imshow(g, 'gray'), plt.title('G')
plt.subplot2grid((3,4),(0,3)), plt.axis('off'), plt.imshow(r, 'gray'), plt.title('R')

plt.subplot2grid((3,4),(1,0)), plt.axis('off'), plt.imshow(src_hsv,'hsv'), plt.title('HSV')
plt.subplot2grid((3,4),(1,1)), plt.axis('off'), plt.imshow(h, 'gray'), plt.title('H')
plt.subplot2grid((3,4),(1,2)), plt.axis('off'), plt.imshow(s, 'gray'), plt.title('S')
plt.subplot2grid((3,4),(1,3)), plt.axis('off'), plt.imshow(v, 'gray'), plt.title('V')

plt.subplot2grid((3,4),(2,0)), plt.axis('off'), plt.imshow(src_ycrcb), plt.title('YCrCb')
plt.subplot2grid((3,4),(2,1)), plt.axis('off'), plt.imshow(y, 'gray'), plt.title('Y')
plt.subplot2grid((3,4),(2,2)), plt.axis('off'), plt.imshow(cr, 'gray'), plt.title('Cr')
plt.subplot2grid((3,4),(2,3)), plt.axis('off'), plt.imshow(cb, 'gray'), plt.title('Cb')

plt.show()