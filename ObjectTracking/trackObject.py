import numpy as np
import cv2

capture = cv2.VideoCapture(0)

# Color range of object in HSV
color_range = [[0,0,0],[179,50,100]]

points = []
count = 0

# Get first frame
response, frame = capture.read()
height, width = frame.shape[:2]

while True:
	response, frame = capture.read()
	frame = cv2.flip(frame, 1)
	
	# converting to hsv
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# threshold values between range
	mask = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))

	# Finding contours
	_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Computing centroids
	centroids = int(height/2), int(width/2)
	radius = 0

	if len(contours) > 0:
		# Find largest contour
		biggest_contour = max(contours, key=cv2.contourArea)
		(x,y), radius = cv2.minEnclosingCircle(biggest_contour)
		M = cv2.moments(biggest_contour)
		try:
			a = int(M['m10']/M['m00'])
			b = int(M['m01']/M['m00'])
			centroids = (a, b) 
		except:
			a = int(height/2)
			b = int(width/2)
			centroids = a,b

		# Threshold contours	
		if radius > 25:
			print (radius)
			cv2.circle(frame, (int(x), int(y)), int(radius), (0,0,255), 2)
			cv2.circle(frame, centroids, 5, (0,255,0), -1)

	points.append(centroids)
	
	# track points
	if radius > 25:
		print (radius)
		for i in range(1, len(points)):
			try:
				cv2.line(frame, points[i-1], points[i], (0,255,0), 2)
			except:
				pass

		count += 0					

	else:
		count += 1	

		# Remove trail when no object in 10 frames
		if count == 10:
			points = []
			count = 0		

	cv2.imshow('Tracked Object', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()