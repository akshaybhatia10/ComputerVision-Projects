import numpy as np
import cv2
from preprocess import *


image = cv2.imread('images/text.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original', image)
cv2.waitKey(0)

cv2.imshow('Gray', gray)
cv2.waitKey(0)

blur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow('Gaussian Blur', blur)
cv2.waitKey(0)

canny = cv2.Canny(blur, 30, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0)

_, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print (len(contours))
# Sort from left to right
contours = sorted(contours, key=contour_coordinates, reverse=False)

# Found number in the image
number = []

for contour in contours:
	(x, y, w, h) = cv2.boundingRect(contour)
	cv2.drawContours(image, contours, -1, (0,255,0), 3)
	cv2.imshow('Contours', image)
	cv2.waitKey(0)

	if w>= 5 and h>=25:
		area = blur[y:y+h, x:x+w]
		res, area = cv2.threshold(area, 127, 255, cv2.THRESH_BINARY_INV)
		

cv2.destroyAllWindows()
