import numpy as np
import cv2

capture = cv2.VideoCapture(0)

# Parameters for kanade optical flow
params_kanade = dict(winSize=(15,15),
 				 maxLevel=2,
  				 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for corner detection
params_corner = dict(maxCorners=100,
		  minDistance = 7,
		  blockSize=7,
		  qualityLevel=0.3)

# Select color for trails
c = np.random.randint(0, 255, (100,3))

# getting corners in first frame
response, first_frame = capture.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_corners = cv2.goodFeaturesToTrack(first_gray, mask = None, **params_corner)

kernel = np.zeros_like(first_frame)

while True:
	response, frame = capture.read()
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# find optical Flow
	corners, status, _ = cv2.calcOpticalFlowPyrLK(first_gray,
												  gray,
												  first_frame_corners, 
												  None, 
												  ** params_kanade)

	# Get good features (old and new)
	new_features = corners[status==1]
	prev_features = first_frame_corners[status==1]

	# draw trails for object tracking
	for i, (new, prev) in enumerate(zip(new_features, prev_features)):
		x0, y0 = new.ravel()
		x1, y1 = prev.ravel()

		kernel = cv2.line(kernel, (x0, y0), (x1, y1), c[i].tolist(), 2)
		frame = cv2.circle(frame, (x0, y0), 5, c[i].tolist(), -1)

	final = cv2.add(frame, kernel)
	
	cv2.imshow('Kanade Optical Flow', final)
	if cv2.waitKey(1) & 0xFF == ord('q'):
			break	

	first_gray = gray.copy()
	first_frame_corners = new_features.reshape(-1, 1, 2)		

cv2.destroyAllWindows()
capture.release()			