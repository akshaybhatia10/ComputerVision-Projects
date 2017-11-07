import cv2
import numpy as np
import dlib

# Path to shape predictor file
PATH = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PATH)
detector = dlib.get_frontal_face_detector() # Return a list of rectangles, corresponding to a face


class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass


def get_landmarks(image):
	points = detector(image, 1)

	if len(points) > 1:
		raise TooManyFaces
	if len(points) == 0:
		return 'error'

	return np.matrix([[t.x, t.y] for t in predictor(image, points[0]).parts()])		

def mark_landmarks(image, landmarks):
	image = image.copy()
	for i, point in enumerate(landmarks):
		position = (point[0,0], point[0,1])
		cv2.putText(image, str(i), position, cv2.FONT_ITALIC, 0.3, (0,0,0), 2)
		cv2.circle(image, position, 2, (0,255,0))

	return image
	
def observe_top(landmarks):
	top_points = []	
	for i in range(50, 53):
		top_points.append(landmarks[i])
	for i in range(61, 64):
		top_points =landmarks[i]
	all_top_points = np.squeeze(np.asarray(top_points))	
	mean_top_points = np.mean(top_points, axis=0)

	return int(mean_top_points[:,1])

def observe_bottom(landmarks):
	bottom_points = []
	for i in range(65,68):
		bottom_points.append(landmarks[i])
	for i in range(56,59):
		bottom_points.append(landmarks[i])
	all_bottom_points = np.squeeze(np.asarray(bottom_points))
	mean_bottom_points = np.mean(bottom_points, axis=0)	

	return int(mean_bottom_points[:,1])

def check_mouth(image):
	landmarks = get_landmarks(image)

	if landmarks == 'error':
		return (image, 0)

	marked_points = mark_landmarks(image, landmarks)
	top = observe_top(landmarks)
	bottom = observe_bottom(landmarks)
	distance = abs(top - bottom)

	return marked_points, distance	
		

capture = cv2.VideoCapture(0)
count = -1
current = False

while True:
	response, frame = capture.read()
	landmarks, mouth_distance = check_mouth(frame)
	previous = current
	if mouth_distance > 25:
		current = True
		cv2.putText(frame, "Still YAWNING", (50,50), cv2.FONT_ITALIC, 2, (0,0,0), 3)
	else:
		current = False
	if previous == True and current == False:
		count += 1

	cv2.putText(frame, "You have yawned {} times".format(str(count+1)), (50,100), cv2.FONT_ITALIC, 1, (0,0,0), 2)	
	#frame = cv2.resize(frame, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_LINEAR)
	cv2.imshow('Landmarks', landmarks)
	cv2.imshow('Yawn Detector', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()		
