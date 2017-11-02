import cv2 
import numpy as np
# HAAR Cascade files
cascade_face = 'cascades/haarcascade_frontalface_default.xml'
cascade_eye = 'cascades/haarcascade_eye.xml'

face_classifier = cv2.CascadeClassifier(cascade_face)
eye_classifier = cv2.CascadeClassifier(cascade_eye)

def face_and_eye_detector(image):

# Reading images and converting to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detecting faces
	faces = face_classifier.detectMultiScale(gray, 1.2, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0), 3)

		# Cropping the face found
		area_gray = gray[y:y+h, x:x+w]
		area_original = image[y:y+h, x:x+w]

		# Detecting eyes
		eyes = eye_classifier.detectMultiScale(area_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(area_original, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
		
	image = cv2.flip(image, 1)		
	return image		

	if faces is ():
		return image		


capture = cv2.VideoCapture(0)

while True:
	response, frame = capture.read()
	cv2.imshow("Live Face and Eye Classifier", face_and_eye_detector(frame))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()