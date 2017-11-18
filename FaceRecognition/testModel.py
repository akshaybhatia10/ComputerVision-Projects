import cv2
import numpy as np
from trainModel import *

cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Finds face and return the original image and cropped face
def detect_face(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if faces is ():
		return image

	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		cropped_face = image[y:y+h, x:x+w]
		cropped_face = cv2.resize(cropped_face, (200,200))
	
	return cropped_face		

capture = cv2.VideoCapture(0)

while True:
	response, frame = capture.read()
	found_face = detect_face(frame)	

	try:
		gray_face = cv2.cvtColor(found_face, cv2.COLOR_BGR2GRAY)

		# predict using trained model
		label, score = classifier.predict(gray_face)
		if score < 500:
			confidence_score = int(100 * (1 - score/400))
			message = 'I am {} confident '.format(str(confidence_score))

		cv2.putText(frame, message, (50,50), cv2.FONT_ITALIC, 1, (200,0,200), 2)

		if score > 75:
			message = 'Sorry! You are not who you say'
			cv2.putText(frame, message, (150,50), cv2.FONT_ITALIC, 1, (200,200,0), 2)
			cv2.imshow('I rekognize You', image)
		else:
			message = 'You you are finally here'
			cv2.putText(frame, message, (150,50), cv2.FONT_ITALIC, 1, (0,200,200), 2)	
			cv2.imshow('I rekognize You', image)
	
	except:
		message = 'No face found!'
		cv2.putText(frame, message, (150,150), cv2.FONT_ITALIC, 1, (0,200,200), 2)	
		cv2.imshow('I rekognize You', frame)
		pass

	if cv2.waitKey(0):
		break

capture.release()
cv2.destroyAllWindows()			

				