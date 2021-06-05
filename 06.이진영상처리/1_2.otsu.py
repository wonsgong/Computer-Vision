import cv2

src = cv2.imread('image/rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

# flag 줄 때 BINARY 는 안써줘도 가능. 단 INV 는 써줘야된다.
th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print("Threshold : ",th)

cv2.imshow('src',src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
