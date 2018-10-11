# import libraries
import cv2
import numpy as np
import dlib

JAW_POINTS = list(range(0, 17))
NOSE_POINTS = list(range(27, 35))
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

ALIGN_POINTS = (LEFT_EYE_POINTS+RIGHT_EYE_POINTS+LEFT_BROW_POINTS+RIGHT_BROW_POINTS+MOUTH_POINTS+NOSE_POINTS)

OVERLAY_POINTS = (LEFT_EYE_POINTS+RIGHT_EYE_POINTS+LEFT_BROW_POINTS+RIGHT_BROW_POINTS+NOSE_POINTS+MOUTH_POINTS)


# Path to shape predictor file
PATH = 'shape_predictor_68_face_landmarks.dat'

# Our landpoints' predictor and detector objects
predictor = dlib.shape_predictor(PATH)
detector = dlib.get_frontal_face_detector()  ##  returns a list of rectangles, each of which corresponding with a face in the image.

# Defining classes for some exception
class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass

# Detect landpoints' on input image
def get_landmarks(image):
	'''
	Returns a 68x2 element matrix, each row of which corresponding with the 
	x, y coordinates of a particular feature point in image.
	'''
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

def convex_hull(image, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(image, points, color=color)


def face_mask(image, landmarks):
	'''
	Generate a mask for the image and a landmark matrix
	'''
	image = np.zeros(image.shape[:2], dtype=np.float64)

	for grp in OVERLAY_POINTS:
		convex_hull(image, landmarks[grp], color=1)

	image = np.array([image, image, image]).transpose((1,2,0))
	image = (cv2.GaussianBlur(image, (11,11), 0) > 0) * 1.0		
	image = cv2.GaussianBlur(image, (11,11), 0)

	return image


def transform_points(p1, p2):
	'''
	Calculates the rotation portion using the Singular Value Decomposition and
	Return the complete transformaton as an affine transformation matrix.
	'''
	p1 = p1.astype(np.float64)
	p2 = p2.astype(np.float64)

	t1 = np.mean(p1, axis=0)
	t2 = np.mean(p2, axis=0)
	p1 -= t1
	p2 -= t2

	s1 = np.std(p1)
	s2 = np.std(p2)
	p1 /= s1
	p2 /= s2

	U, S, V = np.linalg.svd(p1.T * p2)
	R = (U * V).T

	return np.vstack([np.hstack(((s2/s1)*R, t2.T - (s2/s1) * R * t1.T)), np.matrix([0., 0., 1.])])


def read_features(image):
	img = image
	img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
	img = cv2.resize(img, (img.shape[1]* 1, img.shape[0]*1))

	landmarks = get_landmarks(img)

	return img, landmarks

def warp_image(image, M, shape):
	'''
	Maps the second image onto the first and return ithe same
	'''
	initial = np.zeros(shape, dtype=image.dtype)
	cv2.warpAffine(image, M[:2], (shape[1], shape[0]), dst=initial, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)

	return initial


def mix_colors(image1, image2, landmarks, blur_factor=0.6):
	'''
	Changes the colouring of image2 to match that of image1
	'''

	blurred = blur_factor * np.linalg.norm(np.mean(landmarks[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))
	blurred = int(blurred)

	if blurred % 2 == 0:
		blurred += 1
	image1_blur = cv2.GaussianBlur(image1, (blurred, blurred), 0)
	image2_blur = cv2.GaussianBlur(image2, (blurred, blurred), 0)

	image2_blur += (128 * (image1_blur <= 1.0)).astype(image2_blur.dtype)

	return (image2.astype(np.float64) * image1_blur.astype(np.float64) / image2_blur.astype(np.float64))


def swapped(image1 , image2):
	'''
	Combines all function and outputs a swapped image
	'''
	image1, landmarks1 = read_features(image1)
	image2, landmarks2 = read_features(image2)	

	M = transform_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
	
	mask = face_mask(image2, landmarks2)
	warped_mask = warp_image(mask, M, image1.shape)
	combined_mask = np.max([face_mask(image1, landmarks1), warped_mask], axis=0)

	warped_image2 = warp_image(image2, M, image1.shape)
	warped_image2_new = mix_colors(image1, warped_image2, landmarks1)

	final_output = image1 * (1.0 - combined_mask) + warped_image2_new * combined_mask
	cv2.imwrite("SwappedImage3.jpg", final_output)
	return final_output

# Loading our images
image1 = cv2.imread('images/Trump.jpg')
image2 = cv2.imread('images/jongun.jpg')

# face of image2 swapped on image1
swapped_image = swapped(image1, image2)
#cv2.imshow("Swapped 1", swapped_image)	


cv2.waitKey(0)
cv2.destroyAllWindows()








