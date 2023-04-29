import cv2
import numpy as np
import matplotlib.pyplot as plt 
from feature_detection import harris_corner_detection
from feature_descriptor import SIFT_descriptor


#=====================================================
# Attempt - Ransac Method
#=====================================================

def get_homograph_ransac(kp1, kp2, des1, des2):
    
    return None

def SIFT(img):
    # siftDetector= cv2.xfeatures2d.SIFT_create() # limit 1000 points
    siftDetector= cv2.SIFT_create()  # depends on OpenCV version

    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def find_matches2(des1, des2, kp1, kp2):
		'''
		brute force matcher, using L2-distance as metric
		apply thresholding to get best matches
		'''
		# BFMatcher with default params
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2, k=2)

		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append([m])

		matches = []
		for pair in good:
			matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

		matches = np.array(matches)
		return matches

def draw_matches_numpy(img1, img2, matches):
	combo_img = np.concatenate((img2, img1), axis=1)
	offset = combo_img.shape[1]/2
	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.imshow(np.array(combo_img).astype('uint8')) #　RGB is integer type

	ax.plot(matches[:, 0], matches[:, 1], 'xr')
	ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
	ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

	plt.savefig("./../data/output/matched_ransac.jpg")
	
if __name__ == "__main__":
	from feature_detection import harris_corner_detection

	test_img_name1 = "./../data/parrington/prtn01.jpg"
	test_img_name2 = "./../data/parrington/prtn02.jpg"
	# test_img_name1 = "./../data/denny/denny00.jpg"
	# test_img_name2 = "./../data/denny/denny01.jpg"
	img1 = cv2.imread(test_img_name1)
	img2 = cv2.imread(test_img_name2)

	# Our method
	keypoints1 = harris_corner_detection(img1) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint # prtn01.jpg: found 314 keypoints
	keypoints2 = harris_corner_detection(img2) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint # prtn02.jpg: found 305 keypoints

	sift = SIFT_descriptor()
	des1 = sift.cal_descriptor(img1, keypoints1)
	des2 = sift.cal_descriptor(img2, keypoints2)
	
	matches1 = sift.find_matches(des1, des2)    # from two images: 57 matches
	# sift.draw_matches(img1, keypoints1, img2, keypoints2, matches1)


	# CV2 Method
	kp_left, des_left = SIFT(img2)
	kp_right, des_right = SIFT(img1)

    # Convert list of numpy coords, to list of cv2 keypoints
	cv_kp2 = []
	for k in keypoints2:
		y = float(k[0])
		x = float(k[1])
		cv_kp2.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)
	cv_kp1 = []
	for k in keypoints1:
		y = float(k[0])
		x = float(k[1])
		cv_kp1.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)

	print(f"keypoints2:\n{keypoints2}")
	print(f"\ncv_kp2:\n{cv_kp2}")
	print(f"\nkp_left:\n{kp_left}")	

	# matches2 = find_matches2(des_left, des_right, kp_left, kp_right)    # from two images:
	matches2 = find_matches2(des2, des1, cv_kp2, cv_kp1)    # from two images:
	
	draw_matches_numpy(img1, img2, matches2)