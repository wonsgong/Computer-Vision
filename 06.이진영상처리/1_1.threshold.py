import cv2

src = cv2.imread('image/cells.png',cv2.IMREAD_GRAYSCALE)

if src is None :
    print("Image load failed")
    sys.exit()

def onChange(pos) :
    _,dst = cv2.threshold(src, pos, 255,cv2.THRESH_BINARY)
    cv2.imshow('src', dst)

cv2.imshow('src',src)


cv2.createTrackbar('thres', 'src', 0, 255, onChange)
cv2.setTrackbarPos('thres', 'src', 128)


cv2.waitKey()
cv2.destroyAllWindows()