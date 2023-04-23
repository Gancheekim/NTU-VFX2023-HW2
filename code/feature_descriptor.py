import numpy as np
import cv2
import matplotlib.pyplot as plt

class SIFT_descriptor():
	def __init__(self):
		self.bfMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	def rotate_image(self, image, angle, center=None):
		'''
		rotate image according to its center point
		'''
		h, w = image.shape
		if center==None:
			center = (w/2, h/2)
		M = cv2.getRotationMatrix2D(center, angle, 1)
		rotated = cv2.warpAffine(image, M, (w, h))
		return rotated
	
	def normalize(self, v):
		norm = np.linalg.norm(v, ord=2)
		return v/norm if (norm > 0) else v

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
		# this is for finding secondary orientation which has magnitude over 0.8 * main_orientation
		mag_order = np.unique(hist)
		if mag_order[-2] >= 0.8*mag_order[-1]:
			secondary_orientation_bin = np.where(hist == mag_order[-2]) # get the 2nd largest bin index
			secondary_orientation = secondary_orientation_bin[0][0]*10 + 5 # convert back to 360 degree, with +5 offset
			orientations.append(secondary_orientation)
		'''
		return orientations, hist
	
	def cal_descriptor(self, img, keypoints):
		'''
		---- input ----
		img: (H-by-W) array
		keypoints: (N-by-2) array, the 2 are to specify the corresponding y and x coordinate of a keypoint
		---- output ----
		a (N-by-128) array
		'''
		if len(img.shape) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		keypoints_desc = []
		for y, x in keypoints:
			# for each keypoints, obtain its (16x16) nearby patch/window
			img_patch = img[y-8:y+8, x-8:x+8]
			# calculate the main orientation of the patch
			orientations, _ = self.orientation_assignment(img_patch)
			# rotate the patch according to orientation
			img_patch = self.rotate_image(img_patch, orientations[0])

			# for each of the (4x4) size sub-patch, calculate the 8-bin orientation histogram.
			# Thus, we will have 16 sub-patch and 8 orientation for each, total = 128 dimensions.
			desc_128 = []
			for i in range(0, 13, 4):
				for j in range(0, 13, 4):
					img_4x4 = img_patch[i:i+4, j:j+4]
					_, hist = self.orientation_assignment(img_4x4, total_bins=8)
					desc_128 += hist.tolist()
			# normalize the 128 dimension descriptor
			desc_128_arr = np.array(desc_128)
			desc_128_arr = self.normalize(desc_128_arr)
			# clip to 0.2, then normalize again
			if np.any(desc_128_arr > 0.2):
				desc_128_arr[desc_128_arr>0.2] = 0.2
				desc_128_arr = self.normalize(desc_128_arr)
			keypoints_desc.append(desc_128_arr)

		return np.array(keypoints_desc).astype(np.float32) # (N-by-128) array, where N = number of keypoints
	
	def find_matches(self, des1, des2):
		'''
		brute force matcher, using L2-distance as metric
		'''
		return self.bfMatcher.match(des1, des2)
	
	def draw_matches(self, img1, keypoints1, img2, keypoints2, matches):
		'''
		plot the matched points of the 2 imgs
		'''
		cv_kp1 = []
		for k in keypoints1:
			y = float(k[0])
			x = float(k[1])
			cv_kp1.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)

		cv_kp2 = []
		for k in keypoints2:
			y = float(k[0])
			x = float(k[1])
			cv_kp2.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)

		img3 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		cv2.imshow("Matches", img3)
		cv2.waitKey(0)



if __name__ == "__main__":
	from feature_detection import harris_corner_detection

	# test_img_name1 = "./../data/parrington/prtn01.jpg"
	# test_img_name2 = "./../data/parrington/prtn02.jpg"
	test_img_name1 = "./../data/denny/denny00.jpg"
	test_img_name2 = "./../data/denny/denny01.jpg"
	img1 = cv2.imread(test_img_name1)
	img2 = cv2.imread(test_img_name2)

	keypoints1 = harris_corner_detection(img1) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint
	keypoints2 = harris_corner_detection(img2) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint

	sift = SIFT_descriptor()
	des1 = sift.cal_descriptor(img1, keypoints1)
	des2 = sift.cal_descriptor(img2, keypoints2)
	
	matches = sift.find_matches(des1, des2)
	sift.draw_matches(img1, keypoints1, img2, keypoints2, matches)