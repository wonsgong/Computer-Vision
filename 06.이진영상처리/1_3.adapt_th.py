import cv2
import numpy as np
import sys


def onChange(pos) :
    block = pos
    if block % 2 == 0:
        block -= 1
    if block < 3 :
        block = 3

    dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5)
    cv2.imshow('dst', dst)

src = cv2.imread('image/sudoku.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed")
    sys.exit()

cv2.imshow('src', src)
cv2.imshow('dst', src)
cv2.createTrackbar('block', 'dst', 0, 200, onChange)
cv2.setTrackbarPos('block', 'dst',20)



cv2.waitKey()
cv2.destroyAllWindows()
