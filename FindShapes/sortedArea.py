import cv2
import numpy as np

original_image = cv2.imread('images/someshapes.jpg')

gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("Blur", blur)
cv2.waitKey(0)

ret, thresh = cv2.threshold(blur, 127, 255, 1)
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)

canny = cv2.Canny(thresh, 50,200)
cv2.imshow("Canny", canny)
cv2.waitKey(0)

_, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print ('Number of contours', str(len(contours)))

#cv2.drawContours(image, contours, -1, (0,255,0), 3)
#cv2.imshow("Contours on Original", image)
#cv2.waitKey(0)

## Sorting based on areas
def get_contour_areas(contours):
	areas = []
	for contour in contours:
		area = cv2.contourArea(contour)
		areas.append(area)

	return areas

sorted_contours = sorted(contours, key=cv2.contourArea, reverse = True)
print ("Sorted Areas", get_contour_areas(sorted_contours))

for (i, contour) in enumerate(sorted_contours):
	cv2.drawContours(original_image, [contour], -1, (0,255,0), 3)
	M = cv2.moments(contour)
	x = int(M['m10']/ M['m00'])
	y = int(M['m01']/ M['m00'])
	cv2.putText(original_image, str(i+1), (x-10, y+10), cv2.FONT_ITALIC, 1,(255,255,255), 1)
	cv2.waitKey(0)
	cv2.imshow('Contours by area', original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()