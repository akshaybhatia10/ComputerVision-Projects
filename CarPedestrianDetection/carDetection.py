import numpy as np
import cv2
import time

car_cascade = 'cascades/haarcascade_car.xml'
car_classifier = cv2.CascadeClassifier(car_cascade)
capture = cv2.VideoCapture('cars.avi')

while capture.isOpened():

    response, frame = capture.read()
    if response:

    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    	cars = car_classifier.detectMultiScale(gray, 1.2, 3) 

    	for (x, y, w, h) in cars:
    		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 3)
    		cv2.imshow('Cars', frame)

    	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
    else:
    	break    	

        	
capture.release()
cv2.destroyAllWindows()