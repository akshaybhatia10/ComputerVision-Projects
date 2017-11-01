import cv2
import numpy as np

class od_SIFT(object):
	'''
    Object Detection using SIFT (Scale-Invariant Feature Transform)
	'''
	def __init__(self, original_image, template):
		self.original_image = original_image
		self.template = template

	def detector(self):
		'''
		Compares original image with template and finds
		number of SIFT matches
		'''	
		img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
		tem = self.template

		# SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()

		# get keypoints and discriptors (k,d) using sift
		k1, d1 = sift.detectAndCompute(img, None)
		k2, d2 = sift.detectAndCompute(tem, None)

		# flann matcher
		FLANN_INDEX_KDTREE = 0
		index_param = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
		search_param = dict(checks = 100)		

		flann = cv2.FlannBasedMatcher(index_param, search_param)
		#d1, d2 = None, None
		
		# getting all matches using kNN
		matches = flann.knnMatch(d1, d2, k = 2)

		# Save all matches - Lowe's ratio test
		good_matches = []
		for m,n in matches:
			if m.distance < (0.7 * n.distance):
				good_matches.append(m)


		return len(good_matches)
