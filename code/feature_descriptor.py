import numpy as np
import cv2
import matplotlib.pyplot as plt
from feature_detection import harris_corner_detection
from math import tan, pi

class SIFT_descriptor():
	def __init__(self):
		pass

	def rotate_image(self, image, angle, center=None):
		h, w = image.shape
		if center==None:
			center = (w/2, h/2)
		M = cv2.getRotationMatrix2D(center, angle, 1)
		rotated = cv2.warpAffine(image, M, (w, h))
		return rotated
	
	def normalize(self, v):
		norm = np.linalg.norm(v, ord=2)
		if norm == 0: 
			return v
		return v / norm

	def orientation_assignment(self, img_patch, total_bins=36):
		'''
		---- input ----
		img_patch: a (16x16) or (4x4) size of image patch (neighborhood of a keypoint), which will be used to calculate the (16x16) gradient magnitude and orientation
		---- output ----
		orientations: a list, specifying the main (and secondary) orientation of the img_patch, in degree
		hist: a (1x36) array, specifying the histogram of the orientation of img_patch. each bin = 10 degree
		'''
		if len(img_patch.shape) == 3:
			img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
		
		Ix, Iy = np.gradient(img_patch)
		magnitude = np.sqrt(Ix*Ix + Iy*Iy)
		angle = np.arctan2(Iy, Ix) * 180/np.pi 
		angle[angle<0] += 360 # make sure the results are all positive
		bin_interval = 360/total_bins
		angle_bin = np.floor(angle/bin_interval).astype(np.int32)

		# if total_bins = 36, then: 0-9, 10-19, ..., 350-359
		# if total_bins =  8, then: 0-44, 45-89, 90-134, 135-179, ...
		hist = np.zeros((total_bins,))
		H, W = magnitude.shape
		for y in range(H):
			for x in range(W):
				hist[angle_bin[y,x]] += magnitude[y,x]

		orientations = []
		main_orientation_bin = np.argmax(hist, axis=0)
		main_orientation = main_orientation_bin*10 + 5 # convert back to 360 degree, with +5 offset
		orientations.append(main_orientation)

		'''
		mag_order = np.unique(hist)
		if mag_order[-2] >= 0.8*mag_order[-1]:
			secondary_orientation_bin = np.where(hist == mag_order[-2]) # get the 2nd largest bin index
			secondary_orientation = secondary_orientation_bin[0][0]*10 + 5 # convert back to 360 degree, with +5 offset
			orientations.append(secondary_orientation)
		'''
		return orientations, hist
	
	def run(self, img, keypoints):
		if len(img.shape) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		keypoints_desc = []
		for y, x in keypoints:
			img_patch = img[y-8:y+8, x-8:x+8]
			orientations, _ = self.orientation_assignment(img_patch)

			# rotate according to orientation
			img_patch = self.rotate_image(img_patch, orientations[0])
			# for each of the (4x4) size of smaller patch, calculate the 8-bin orientation histogram
			desc_128 = []
			for i in range(0, 13, 4):
				for j in range(0, 13, 4):
					img_4x4 = img_patch[i:i+4, j:j+4]
					_, hist = self.orientation_assignment(img_4x4, total_bins=8)
					desc_128 += hist.tolist()
			# normalize
			desc_128_arr = np.array(desc_128)
			desc_128_arr = self.normalize(desc_128_arr)
			# clip to 0.2, then normalize again
			if np.any(desc_128_arr > 0.2):
				desc_128_arr[desc_128_arr>0.2] = 0.2
				desc_128_arr = self.normalize(desc_128_arr)
			desc_128 = desc_128_arr.tolist()
			# keypoints_desc.append({'point':(y,x), 'desc':desc_128})
			keypoints_desc.append(desc_128)

		return keypoints_desc



if __name__ == "__main__":
	'''
	test_img_name = "./../data/grail/grail01.jpg"
	img = cv2.imread(test_img_name)
	keypoints = harris_corner_detection(img)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	descriptor = SIFT_descriptor()
	keypoints_desc = descriptor.run(img, keypoints)
	'''

	test_img_name = "./../data/parrington/prtn01.jpg"
	img1 = cv2.imread(test_img_name)

	test_img_name = "./../data/parrington/prtn02.jpg"
	img2 = cv2.imread(test_img_name)

	keypoints1 = harris_corner_detection(img1) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint
	keypoints2 = harris_corner_detection(img2) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint

	# print(type(keypoints1))

	# Detect keypoints in both images using the SIFT detector
	sift = SIFT_descriptor()
	des1 = sift.run(img1, keypoints1)
	des2 = sift.run(img2, keypoints2)
	des1 = np.array(des1).astype(np.float32)
	des2 = np.array(des2).astype(np.float32)

	# Convert the SIFT keypoints to cv2.KeyPoint objects
	cv_kp1 = []
	for k in keypoints1:
		# (x, y, size, angle, response, octave, class_id)
		cv_kp1.append(cv2.KeyPoint(k[0], k[1], _size=k[2], _angle=k[3], _response=k[4], _octave=k[5], _class_id=k[6]))

	cv_kp2 = []
	for k in keypoints2:
		# (x, y, size, angle, response, octave, class_id)
		cv_kp2.append(cv2.KeyPoint(k[0], k[1], _size=k[2], _angle=k[3], _response=k[4], _octave=k[5], _class_id=k[6]))

	# print(des1.dtype)
	# print(des2.dtype)

	# Create a brute-force matcher object
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	# # Match the keypoints in both images
	matches = bf.match(des1, des2)
	# print(matches)

	# Visualize the matches
	img3 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv2.imshow("Matches", img3)
	cv2.waitKey(0)
