import cv2
import numpy as np

# HaarCascade face classifier
cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def get_face(image):
	'''
	Returns face found in the image(or video frame)
	'''
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if faces is ():
		return None

	for (x, y, w, h) in faces:
		cropped_face = image[y:y+h, x:x+w]

	return (x,y,w,h), cropped_face		

capture = cv2.VideoCapture(0)
num_faces = 0
new_dimension = (200, 200)

while True:
	response, frame = capture.read()
	if get_face(frame) is not None:
		num_faces += 1

		(x,y,w,h), found_face = get_face(frame)
		# resizing found face to new dimensions
		resized_face = cv2.resize(found_face, new_dimension)
		# Converting to grayscale again
		resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)	

		# Saving the training data
		path = 'trainingData/' + str(num_faces) + '.jpg'
		cv2.imwrite(path, resized_face)

		# To track number of images completed
		cv2.putText(resized_face, str(num_faces), (50,50), cv2.FONT_ITALIC, 1, (0,0,0), 2)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,250,0), 2)
		cv2.imshow('Saving training data', frame)
		cv2.imshow("Found face", resized_face)

	else:
		print ("No face found")
		pass

	if cv2.waitKey(1) == 13 or  num_faces == 50:
		break

capture.release()
cv2.destroyAllWindows()

print ('Training data done')				