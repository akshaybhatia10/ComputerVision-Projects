import numpy as np
import cv2

original_image = cv2.imread('images/someshapes.jpg')
print("Shape: Original Image", original_image.shape)

# Converting to grayscale
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
print("Shape: Grayscaled Image",gray.shape)
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)

# Thresholding (Optional)
ret, thresh = cv2.threshold(gray, 127, 255, 1)
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print ('Number of contours', str(len(contours)))

# Draw contours on original image
#cv2.drawContours(original_image, contours, -1, (0,255,0), 3)
#cv2.imshow("Contours on Original", original_image)
#cv2.waitKey(0)

cv2.destroyAllWindows()