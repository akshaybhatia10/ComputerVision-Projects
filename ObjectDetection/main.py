import cv2
import numpy as np

# importing SIFT detector
from SIFT_detector import od_SIFT

# importing ORB detector
from ORB_detector import od_ORB

# Loading original template image
img = 'images/phone.png'
template = cv2.imread(img)
cv2.imshow("Template Image", template)
cv2.waitKey(0)

# Initialize videoCapture
capture = cv2.VideoCapture(0)

# Choose the type of object detection algorithm
# od_ORB: ORB (Oriented FAST and Rotated BRIEF)
# od_SIFT: SIFT (Scale-Invariant Feature Transform)
use_this = od_ORB


while True:
	# image from webcam
	response, frame = capture.read()

	# height and width of webcam images
	height, width = frame.shape[:2]

	# Box dimensions 
	top_x = int(width/3)
	top_y = int((height/2) + (height/4))
	bottom_x = int((width/3) * 2)
	bottom_y = int((height/2) - (height/4))

	# drawing the rectangle with above points
	cv2.rectangle(frame, (top_x, top_y), (bottom_x, bottom_y), (0,255,0), 3)

	# Seperate the above area
	area_of_interest = frame[bottom_y:top_y, top_x:bottom_x]

	# flipping the frame
	frame = cv2.flip(frame, 1)

	# Defining object for 'od_SIFT' class
	f = use_this(area_of_interest, template)
	matches = f.detector()

	# Updating results and showing result on screen
	text = ("We have {} matches".format(str(matches)))
	cv2.putText(frame, text, (300,630), cv2.FONT_ITALIC, 2 ,(0,0,0), 8)

	# threshold to show object detection
	threshold = 500

	if matches > threshold:
		cv2.rectangle(frame, (top_x, top_y), (bottom_x, bottom_y), (255,255,0), 3)
		result = "WOW!! Found!"
		cv2.putText(frame, result, (750,50), cv2.FONT_ITALIC, 2, (0,255,0), 2)

	cv2.imshow("OBJECT DETECTION USING SIFT", frame)	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


capture.release()
cv2.destroyAllWindows()