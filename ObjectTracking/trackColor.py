import cv2
import numpy as np


capture = cv2.VideoCapture(0)

color_range = [[0,0,0], [0,0,255]]

while True:
	response, frame = capture.read()

	# BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# filter values between the specified range
	filter_mask = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))

	# Bitwise AND on filter mask and frame
	color_detected = cv2.bitwise_and(frame, frame, mask=filter_mask)

	cv2.imshow('Original', frame)
	cv2.imshow('Filtered', filter_mask)
	cv2.imshow('Detected Object using color', color_detected)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


capture.release()
cv2.destroyAllWindows()		