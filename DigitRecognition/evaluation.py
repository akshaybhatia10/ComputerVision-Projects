import numpy as np
import cv2
from preprocess import *
from basic_knn import *

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

_, contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print (len(contours))
# Sort from left to right
contours = sorted(contours, key=contour_coordinates, reverse=False)

# Found number in the image
display = []

for contour in contours:
	(x, y, w, h) = cv2.boundingRect(contour)
	#cv2.drawContours(image, contours, -1, (0,255,0), 3)
	#cv2.imshow('Contours', image)
	#cv2.waitKey(0)

	if w>= 5 and h>=20:
		area = blur[y:y+h, x:x+w]
		ret, area = cv2.threshold(area, 127, 255, cv2.THRESH_BINARY_INV)

		new_square = drawSquare(area)
		number = resize(new_square, 20)
		cv2.imshow('Numbers', number)
		cv2.waitKey(0)
		print (number.shape)
		result = number.reshape((1, 400))
		result = number.astype(np.float32)
		ret, res, neighbours, distance = classifier_knn.findNearest(result, k=1)
		n = str(int(float(res[0])))
		display.append(n)

		# draw rectangle around individual digit
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
		cv2.putText(image, n, (x,y+155), cv2.FONT_ITALIC, 2, (0,0,130), 2)
		cv2.imshow('Image with numbers decoded', image)
		cv2.waitKey(0)

cv2.destroyAllWindows()
