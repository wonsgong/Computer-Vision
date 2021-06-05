import sys
import time
import numpy as np
import cv2


img = cv2.imread('hongkong.jpg')

tm = cv2.TickMeter()

tm.reset()
tm.start()

edge = cv2.Canny(img, 50, 150)

tm.stop()
print('Elapsed time: {}ms.'.format(tm.getTimeMilli()))

