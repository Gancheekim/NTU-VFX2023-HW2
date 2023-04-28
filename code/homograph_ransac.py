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

# REFERENCE
def compute_error(homograph, matches):
    num_points = len(matches)     # number of matches
    all_p1 = np.concatenate((matches[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = matches[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(homograph, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors


# Solve homograph
def get_homograph_ransac(matches, num_iters=2000, num_samples=4, threshold=0.5):
	# print(matches)  					 		     # [[299.0, 162.0, 51.0, 160.0], [291.0, 191.0, 43.0, 186.0], [319.0, 194.0, 78.0, 175.0], [370.0, 170.0, 121.0, 168.0]]
	matches_arr = np.array(matches)
	num_best_inliers = 0  # score to maximize

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
		M = []    # matrix representation to do SVD
		for match_pair in sample_matches:
			p1 = match_pair[:2] + [1]    # e.g [370.0, 170.0, 1]  
			p2 = match_pair[2:] + [1]	 # e.g [121.0, 168.0, 1]
			
			# Construct Matrix
			row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]    # [0, 0, 0, 370.0, 170.0, 1, -62160.0, -28560.0, -168.0]
			row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]    # [370.0, 170.0, 1, 0, 0, 0, -44770.0, -20570.0, -121.0]
			M.append(row1)
			M.append(row2)

			# print(f"p1:\n{p1}\n")
			# print(f"p2:\n{p2}\n")
			# print(f"row1:\n{row1}\n")
			# print(f"row2:\n{row2}\n")
			# print(M)

		# Solve Matrix
		M = np.array(M)
		U, S, Vh = np.linalg.svd(M)
		homograph = Vh[-1].reshape(3, 3)
		homograph = homograph / homograph[2,2]  # normalize
		#print(f"homograph:\n{homograph}")

		#-------------------------------------------------------------------------------
		# Step 3: 
		#-------------------------------------------------------------------------------
		#  Skip iteration if rank < 3
		if np.linalg.matrix_rank(homograph) < 3:
			continue

        # Find matches with errors < threshold   
		errors = compute_error(homograph, matches_arr)
		best_idx = np.where(errors < threshold)[0]    # threshold = 0.75
		inliers = matches_arr[best_idx]
		#print(f"errors:\n{errors}")

		# Record best iteration results
		num_inliers = len(inliers)
		if num_inliers > num_best_inliers:
			best_inliers = inliers.copy()
			num_best_inliers = num_inliers
			best_H = homograph.copy()

	print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
	
	return best_inliers, best_H



def plot_matches(matches, img_left, img_right):
    total_img = np.concatenate((img_left, img_right), axis=1)
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(total_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]], 'r', linewidth=0.5)

    plt.savefig("./../data/output/ransac_output.png")

if __name__ == "__main__":
	from feature_detection import harris_corner_detection

	# Read images
	test_img_name_right = "./../data/parrington/prtn01.jpg"  # "./../data/denny/denny01.jpg"
	test_img_name_left  = "./../data/parrington/prtn02.jpg"  # "./../data/denny/denny00.jpg"
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
	sift = SIFT_descriptor()										   							# declare SIFT class object
	des_right = sift.cal_descriptor(img_right, coord_right)			   							# get descriptor for right image
	des_left  = sift.cal_descriptor(img_left, coord_left)			   							# get descriptor for left image

	#==================
	# OpenCV Method
	#==================
	'''
	Comment the following 3 lines to use our method
	'''
	# siftDetector= cv2.SIFT_create()
	# cv_kp_left, des_left = siftDetector.detectAndCompute(img_left, None)
	# cv_kp_right, des_right = siftDetector.detectAndCompute(img_right, None)


	# Basic filtering out match outliers
	init_matches = sift.find_matches(des_left, des_right)    		   						    # get initial matches by BruteForce: 57 matches
	good, matches_list = filter_matches(cv_kp_left, cv_kp_right, init_matches, threshold=0.9)   # get good matches

	# Draw Match Results: cv.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(img_left, cv_kp_left, img_right, cv_kp_right, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv2.imwrite("./../data/output/matched_precleaned.jpg", img3)

	# RANSAC
	best_inliers, best_homograph = get_homograph_ransac(matches_list)
	print(matches_list)
	print(best_inliers)
	plot_matches(best_inliers, img_left, img_right)
