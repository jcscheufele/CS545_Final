import cv2 as cv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
car = cv.imread("../Images/01.jpg")
target = cv.imread("../Images/01-t.jpg", 0)

cv.imwrite("../Images/gray-t.jpg", target)
print(target[350])
print(target.shape)

target[target <= 100] = 1
target[target > 100] = 0
target[target == 1] = 255
print(target.shape)

cv.imwrite("../Images/gray-flip-t.jpg", target)
print(target[350])