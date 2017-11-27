import numpy as np
import cv2

capture = cv2.VideoCapture(0)

# Get initial frame
response, first_frame = capture.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255

while True:
	response, frame = capture.read()
	frame = cv2.flip(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Calculates dense optical flow
	dense_flow = cv2.calcOpticalFlowFarneback(first_gray,
											  frame, 
											  None,
											  0.5, 3, 15, 3, 5, 1.2, 0)

	# Calculate speed and angle(theta) of motion
	speed, theta = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
	hsv[..., 0] = theta * (180 / (np.pi/2))
	hsv[..., 2] = cv2.normalize(speed, None, 0, 255, cv2.NORM_MINMAX)
	final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	cv2.imshow('Dense Optical Flow', final)
	if cv2.waitKey(1) & 0xFF == ord('q'):
			break	

	first_gray = frame		

cv2.destroyAllWindows()
capture.release()			