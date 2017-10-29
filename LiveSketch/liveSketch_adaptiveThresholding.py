import numpy as np
import cv2

def sketch(frame):
	'''
	Generate sketch given an image
	@paramaters: frame 
	'''
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	canny = cv2.Canny(blur, 10, 70)
	#lap = cv2.Laplacian(blur, cv2.CV_8UC1)

	# Adaptive Thresholding - No need to spicify threshold value
	thresh = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
	                               cv2.THRESH_BINARY, 3, 5)
	return thresh

capture = cv2.VideoCapture(0)

while (True):
	response, frame = capture.read()
	cv2.imshow("Those edges(Adaptive Thresholding)", sketch(frame))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()	