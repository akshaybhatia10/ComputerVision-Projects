import cv2
import dlib
import numpy as np

# Path to shape predictor file
PATH = 'shape_predictor_68_face_landmarks.dat'

# Our landpoints' predictor and detector objects
predictor = dlib.shape_predictor(PATH)
detector = dlib.get_frontal_face_detector() 

# Reading our image
img = cv2.imread('images/mr_robot.jpg')

# Defining classes for some exception
class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass

# Detect landpoints' on input image
def get_landmarks(image):
	points = detector(image, 1)

	if len(points) > 1:
		raise TooManyFaces
	if len(points) == 0:
		raise NoFaces

	return np.matrix([[t.x, t.y] for t in predictor(image, points[0]).parts()])


# Mark and point landmarks' on input image using numbers
def mark_landmarks(image, landmarks):
	image = image.copy()
	for i, point in enumerate(landmarks):
		position = (point[0,0], point[0,1])
		cv2.putText(image, str(i), (position), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(0,0,0))
		cv2.circle(image, position, 3, color=(0,255,0))

	return image


l = get_landmarks(img)
marked_image = mark_landmarks(img, l)

cv2.imshow("Landmarks", marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()					
