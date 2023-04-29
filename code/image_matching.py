import cv2
import numpy as np
import matplotlib.pyplot as plt 
from feature_detection import harris_corner_detection
from feature_descriptor import SIFT_descriptor
from cylindrical_proj import cylindrical_proj

# def find_matches2(des1, des2, kp1, kp2, thresh=0.8):
# 		'''
# 		brute force matcher, using L2-distance as metric
# 		apply thresholding to get best matches
# 		'''
# 		# BFMatcher with default params
# 		bf = cv2.BFMatcher()
# 		matches = bf.knnMatch(des1,des2, k=2)

# 		# Apply ratio test
# 		good = []
# 		for m,n in matches:
# 			if m.distance < thresh*n.distance:
# 				good.append([m])

# 		matches = []
# 		for pair in good:
# 			matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt)) # [x1, y1, x2, y2]

# 		matches = np.array(matches)
# 		return matches

def filter_matches(des1, des2, kp_left, kp_right, threshold=0.8):
	'''
	brute force matcher, using L2-distance as metric
	apply thresholding to get best matches
	'''
	# BFMatcher with default params
	bf = cv2.BFMatcher()
	init_matches = bf.knnMatch(des1,des2, k=2)

	# Thresholding to select good matches from initial matches
	good = [[m] for m,n in init_matches if m.distance < threshold*n.distance]

	# Obtain matches list
	matches_list = [list(kp_left[pair[0].queryIdx].pt + kp_right[pair[0].trainIdx].pt) for pair in good]

	# Return in array format
	return np.array(matches_list)

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

	cv2.imwrite("./../data/output/matched_ransac.jpg", total_img)


def ransac_img_matching(matches, img1, img2, wind_sz=400, blursigma=1, x_offset_lim=2, y_offset_lim=50):
	'''
	----- input -----
	matches: a N-by-4 array, each entries specifies [x1, y1, x2, y2]
	img1: the image at the left-hand-side
	img2: the image at the right-hand-side
	'''
	best_match = []
	best_diff = 2**30
	H1,W1,_ = img1.shape
	H2,W2,_ = img2.shape
	limit_region = int(W2/x_offset_lim)
	for x1, y1, x2, y2 in matches:
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		if x2 >= limit_region or x1 <= W1-limit_region or abs(y2-y1) >= y_offset_lim: 
			continue
		
		left_width = min(x2, x1)
		right_width = min(W1-x1, W2-x2)
		up_height = min(y1, y2)
		low_height = min(H1-y1, H2-y2) - 1

		window1 = img1[y1-up_height:y1+low_height, x1-left_width:x1+right_width, :]
		window2 = img2[y2-up_height:y2+low_height, x2-left_width:x2+right_width, :]

		window1 = cv2.GaussianBlur(window1, (3,3), blursigma)
		window2 = cv2.GaussianBlur(window2, (3,3), blursigma)

		H,W,_ = window1.shape
		random_points_x = np.random.randint(W-1, size=wind_sz)
		random_points_y = np.random.randint(H-1, size=wind_sz)

		random_points1 = window1[random_points_y, random_points_x, :]
		random_points2 = window2[random_points_y, random_points_x, :]

		diff = np.sum(np.absolute(random_points1 - random_points2))
		if diff < best_diff:
			# print(f'best diff: (x1,y1): ({x1},{y1}), (x2,y2): ({x2},{y2})')
			best_diff = diff
			best_match = [x1, y1, x2, y2]

	offset_y = best_match[3] - best_match[1]
	W = img1.shape[1]
	offset_x = -best_match[2] - W + best_match[0]

	return offset_x, offset_y


def draw_stitch_result(off_x, off_y, img1, img2):
	print(f'{off_x}, {off_y}')

	h,w,c = img1.shape
	plt.figure()
	pano = np.zeros((h, w*2, 3))
	pano[:, :w, :] = img1

	w1 = int(w+off_x)
	w2 = int(w1+w)
	if off_y < 0:
		pano[-off_y:, w1:w2 , :] = img2[:off_y, :, :]
	elif off_y > 0:
		pano[:-off_y, w1:w2 , :] = img2[off_y:, :, :]
	else:
		pano[:, w1:w2 , :] = img2[:, :, :]
	plt.imshow(pano/255)
	plt.savefig("./../data/output/stitch2.jpg")
	plt.show()


if __name__ == "__main__":
	# test_img_name1 = "./../data/parrington/prtn06.jpg" # left
	# test_img_name2 = "./../data/parrington/prtn05.jpg" # right
	# test_img_name1 = "./../data/parrington/prtn07.jpg" # left
	# test_img_name2 = "./../data/parrington/prtn06.jpg" # right
	# test_img_name1 = "./../data/parrington/prtn02.jpg" # left
	# test_img_name2 = "./../data/parrington/prtn01.jpg" # right
	test_img_name1 = "./../data/parrington/prtn01.jpg" # left
	test_img_name2 = "./../data/parrington/prtn00.jpg" # right
	img1 = cv2.imread(test_img_name1)
	img2 = cv2.imread(test_img_name2)

	thresh = 0.75

	# img1 = cylindrical_proj(img1, 705.327)
	# img2 = cylindrical_proj(img2, 705.645)
	# img1 = cylindrical_proj(img1, 704.696)
	# img2 = cylindrical_proj(img2, 705.327)
	# img1 = cylindrical_proj(img1, 705.849)
	# img2 = cylindrical_proj(img2, 706.286)
	img1 = cylindrical_proj(img1, 706.286)
	img2 = cylindrical_proj(img2, 704.916)

	# Our method
	keypoints1 = harris_corner_detection(img1) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint # prtn01.jpg: found 314 keypoints
	keypoints2 = harris_corner_detection(img2) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint # prtn02.jpg: found 305 keypoints

	sift = SIFT_descriptor()
	des1 = sift.cal_descriptor(img1, keypoints1)
	des2 = sift.cal_descriptor(img2, keypoints2)

	# Convert list of numpy coords, to list of cv2 keypoints
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
	
	matches2 = filter_matches(des1, des2, cv_kp1, cv_kp2, thresh) # [x1, y1, x2, y2]

	draw_matches(img1, img2, matches2)

	off_x, off_y = ransac_img_matching(matches2, img1, img2)

	print(f'{off_x}, {off_y}')

	h,w,c = img1.shape
	plt.figure()
	pano = np.zeros((h, w*2, 3))
	pano[:, :w, :] = img1

	w1 = int(w+off_x)
	w2 = int(w1+w)
	if off_y < 0:
		pano[-off_y:, w1:w2 , :] = img2[:off_y, :, :]
	elif off_y >= 0:
		pano[:-off_y, w1:w2 , :] = img2[off_y:, :, :]
	plt.imshow(pano/255)
	plt.show()