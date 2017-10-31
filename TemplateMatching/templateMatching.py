import cv2
import numpy as np

# Loading original template image
original_image_template = cv2.imread('images/waldo.jpg')
cv2.imshow("Original Template", original_image_template)
cv2.waitKey(0)

# Getting Dimensions to draw box similar to template
height, width = original_image_template.shape[:2]

original_image = cv2.imread('images/WaldoBeach.jpg')
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)

# Converting to grayscale
gray_template = cv2.cvtColor(original_image_template, cv2.COLOR_BGR2GRAY)
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Matching template with original image
match = cv2.matchTemplate(gray_original, gray_template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

# drawing rectangle around the matched area
top_left = max_loc
bottom_right = (top_left[0]+height, top_left[1]+width)
cv2.rectangle(original_image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow("Original Image with matched area", original_image)
cv2.waitKey(0)

cv2.destroyAllWindows()