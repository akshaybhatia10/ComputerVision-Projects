import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

class od_ORB(object):
	"""
    Object Detection using ORB (Oriented FAST and Rotated BRIEF)
	"""
	def __init__(self, original_image, template):
		self.original_image = original_image
		self.template = template

	def detector(self):
		'''
		Compares original image with template and finds
		number of ORB matches
		'''	
		img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
		tem = self.template

		# SIFT detector - 1000 keypoints, scale 1.2
		orb = cv2.ORB_create(1000, 1.2)

		# get keypoints and discriptors (k,d) using sift
		k1, d1 = orb.detectAndCompute(img, None)
		k2, d2 = orb.detectAndCompute(tem, None)

		# Defining matcher for ORB
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

		# getting all matches 
		matches = matcher.match(d1, d2)

		# Save all matches - sort in ascending order
		matches = sorted(matches, key=lambda val: val.distance)

		return len(matches)
