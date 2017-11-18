import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

# Extract all images from the file path
path = 'trainingData/'
files = [image for image in listdir(path) if isfile(join(path, image))]

inputs = []
targets = []

for i, file in enumerate(files):
	image_path = path + files[i]
	print (image_path)
	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	inputs.append(np.asarray(img, dtype=np.uint8))
	targets.append(i)

targets = np.asarray(targets, dtype=np.int32)

classifier = cv2.face.createLBPHFaceRecognizer()	 

classifier.train(np.asarray(inputs), np.asarray(targets))

print ('Done.')