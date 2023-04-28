import cv2
import numpy as np
import matplotlib.pyplot as plt 
from cylindrical_proj import cylindrical_proj
from feature_detection import harris_corner_detection
from feature_descriptor import SIFT_descriptor

from random import sample

#=====================================================
# Attempt - Ransac Method
#=====================================================
# Pre-process Matches
def coord_to_keypoints(coord_list):
	cv_kp = []
	for coord in coord_list:
		y = float(coord[0])
		x = float(coord[1])
		cv_kp.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)
	return cv_kp

def filter_matches(kp_left, kp_right, init_matches, threshold):
		'''
		brute force matcher, using L2-distance as metric
		apply thresholding to get best matches
		'''
		# Thresholding to select good matches from initial matches
		good = [[m] for m,n in init_matches if m.distance < threshold*n.distance]

		# Obtain matches list
		matches_list = [list(kp_left[pair[0].queryIdx].pt + kp_right[pair[0].trainIdx].pt) for pair in good]

		# Return in array format
		return good, matches_list


def compute_error(homograph, matches):
	num_matches = len(matches)    # number of matches e.g 229
	
	# Separate points from left image, and points from right image
	U = np.array([list(m[ :2]) + [1] for m in matches]) # U is the set of points u in the left image; append 1 to the back for homogeneity
	V = np.array([list(m[2: ]) for m in matches]) 		# V is the set of points v in the left image

	# Estimate of a points V in second image, using homograph and U
	est_V = np.zeros(shape=(num_matches, 2))  # there are 'number of matches' of v, each point has 2 coordinate values
	for i in range(num_matches):
		dot_result = np.dot(homograph, U[i])  # numpy dot product
		v          = (dot_result/dot_result[2])[:2]   # get the two coordinate values for v
		est_V[i]   = v.copy()
	
	# Compute Error (V vs. estimated V)
	error = np.linalg.norm(V - est_V, axis=1)
	return error

	


# Solve homograph
def get_homograph_ransac(matches, num_iters=1500, num_samples=4, threshold=0.5):		 		     
	matches_arr  = np.array(matches)    # [[299.0, 162.0, 51.0, 160.0], [291.0, 191.0, 43.0, 186.0], [319.0, 194.0, 78.0, 175.0], [370.0, 170.0, 121.0, 168.0]]
	best_score   = 0  				    # number of inliers matches, the score to maximize by RANSAC
	best_inliers = []                	# inlier matches record
	# Do ransac iteration
	for i in range(num_iters):
		#-------------------------------------------------------------------------------
		# Step 1: Randomly sample points from matches to determine model parameters
		#-------------------------------------------------------------------------------
		sample_matches = sample(matches, num_samples)    # (min. num. of points = 4 in each sample)
		#print("\nsample_matches: ", sample_matches)     # [[370.0, 170.0, 121.0, 168.0], [283.0, 14.0, 208.0, 65.0], [233.0, 444.0, 75.0, 478.0], [319.0, 194.0, 78.0, 175.0], [270.0, 258.0, 171.0, 404.0], [291.0, 191.0, 43.0, 186.0]]

		#-------------------------------------------------------------------------------
		# Step 2: Determine homography of two images, based on sampled points
		#-------------------------------------------------------------------------------
		# Construct Matrix to Solve by SVD
		M = np.zeros(shape=(num_samples*2, 9))   # matrix intialization, num_samples*2 rows, 3x3 values in each row
		for match_idx, match_pair in enumerate(sample_matches):
			# Identify the coordinates of two points matched
			u, v = match_pair[:2], match_pair[2:]     # e.g [370.0, 170.0]  (y1,x1) from image 1, [121.0, 168.0]  (y2,x2) from image 2

			# Sub Matrix (every match pair will contribute 2 rows to the matrix)
			sub_matrix = [
				[u[0], u[1], 1, 0, 0, 0, -u[0]*v[0], -u[1]*v[0], -1*v[0]],    # [370.0, 170.0, 1, 0, 0, 0, -44770.0, -20570.0, -121.0]
				[0, 0, 0, u[0], u[1], 1, -u[0]*v[1], -u[1]*v[1], -1*v[1]],    # [0, 0, 0, 370.0, 170.0, 1, -62160.0, -28560.0, -168.0]
			]

			# Fill in the matrix with sub matrices
			M[match_idx*2]   = sub_matrix[0]    # fill in matrix rows 0,2,4,6
			M[match_idx*2+1] = sub_matrix[1]    # fill in matrix rows 1,3,5,7

		# Solve Matrix
		U, S, Vh = np.linalg.svd(M)			    # perform single value decomposition
		homograph = Vh[-1].reshape(3, 3)        # obtain a 9x9 matrix
		homograph = homograph / homograph[2,2]  # normalize to satisfy homogeneity
		#print(f"homograph:\n{homograph}")

		#-------------------------------------------------------------------------------
		# Step 3: 
		#-------------------------------------------------------------------------------
		#  Skip iteration if rank < 3
		if np.linalg.matrix_rank(homograph) < 3: continue

        # Find matches with errors < threshold, these are the inliers
		error = compute_error(
			homograph = homograph,
			matches   = matches_arr,
			)

		best_idx = np.where(error < threshold)[0]    # get indeces, default threshold = 0.6
		inliers = matches_arr[best_idx]
		#print(f"errors:\n{errors}")

		# Record best iteration results
		score = len(inliers)					      # score for current iteration of RANSAC algorithm
		if score > best_score:						  # if new best found 
			best_score     = score				      # update score
			best_inliers   = inliers				  # update inliers recorded
			best_homograph = homograph				  # update homograph recorded

	return best_score, best_inliers, best_homograph

def draw_matches(img_left, img_right, matches):
	total_img = cv2.hconcat([img_left, img_right])
	W = total_img.shape[1]  					 		# since we concatenated the image, we should add offset to the x-coordinates
	offset = W/2								 		# offset in x-direction
	
	for i, m in enumerate(matches):
		color = (255, 255, 0) 											# in current colorspace, this is light blue
		u, v = (int(m[0]), int(m[1])), (int(m[2] + offset), int(m[3]))  # coordinates of two points in this match
		# Draw line
		cv2.line(
			img = total_img,
			pt1 = u,
			pt2 = v,
			color = color,  
			thickness = 1,
		)
		# Draw markers
		cv2.drawMarker(
			img = total_img,
			position = u,
			color = color,
			markerType = cv2.MARKER_SQUARE,
			markerSize = 10,
			thickness = 1,
			line_type = 8,	
		)
		cv2.drawMarker(
			img = total_img,
			position = v,
			color = color,
			markerType = cv2.MARKER_SQUARE,
			markerSize = 10,
			thickness = 1,
			line_type = 8,	
		)

	return total_img


if __name__ == "__main__":
	from feature_detection import harris_corner_detection

	# Read images
	test_img_name_right = "./../data/parrington/prtn12.jpg"  # "./../data/denny/denny01.jpg"
	test_img_name_left  = "./../data/parrington/prtn13.jpg"  # "./../data/denny/denny00.jpg"
	img_right = cylindrical_proj(cv2.imread(test_img_name_right), 704.676)
	img_left  = cylindrical_proj(cv2.imread(test_img_name_left), 704.676)
	# img_right = cv2.imread(test_img_name_right)
	# img_left  = cv2.imread(test_img_name_left)

	#==================
	# Our Method
	#==================
	# Get feature points (corner)
	coord_right = harris_corner_detection(img_right) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint # prtn01.jpg: found 314 keypoints
	coord_left  = harris_corner_detection(img_left)  # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint # prtn02.jpg: found 305 keypoints
	
	# Convert list of numpy coords, to list of cv2 keypoints
	cv_kp_right = coord_to_keypoints(coord_right)
	cv_kp_left  = coord_to_keypoints(coord_left)

	# Use SIFT
	sift = SIFT_descriptor()								 # declare SIFT class object
	des_right = sift.cal_descriptor(img_right, coord_right)	 # get descriptor for right image
	des_left  = sift.cal_descriptor(img_left, coord_left)	 # get descriptor for left image

	#==================
	# OpenCV Method
	#==================
	'''
	Comment the following 3 lines to use our method
	'''
	siftDetector= cv2.SIFT_create()
	cv_kp_left, des_left = siftDetector.detectAndCompute(img_left, None)
	cv_kp_right, des_right = siftDetector.detectAndCompute(img_right, None)


	# Basic filtering out match outliers
	init_matches = sift.find_matches(des_left, des_right)    		   						    # get initial matches by BruteForce: 57 matches
	good, matches_list = filter_matches(cv_kp_left, cv_kp_right, init_matches, threshold=0.7)   # get good matches

	# Draw Match Results: cv.drawMatchesKnn expects list of lists as matches.
	# Precleaning of outlier matches
	img3 = cv2.drawMatchesKnn(img_left, cv_kp_left, img_right, cv_kp_right, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv2.imwrite("./../data/output/matched_precleaned.jpg", img3)

	# RANSAC
	best_score, best_inliers, best_homograph = get_homograph_ransac(
		matches     = matches_list,
		num_iters   = 1,
		num_samples = 4,
		threshold   = 0.5
		)
	img4 = draw_matches(img_left, img_right, best_inliers)
	cv2.imwrite("./../data/output/ransac_output.jpg", img4)
