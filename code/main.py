from cylindrical_proj import cylindrical_proj
from feature_detection import harris_corner_detection
from feature_descriptor import SIFT_descriptor
from image_matching import filter_matches, ransac_img_matching, draw_matches, draw_stitch_result
from stitch_panorama import Panorama

from argparse import ArgumentParser
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def read_data(path):
	img_list = []
	focal_len_list = []
	filename_list = []
	for file in os.listdir(path):
		filename_list.append(file)
	if "parrington" in path:
		filename_list.sort(reverse=True)

	for file in filename_list:
		if file.endswith(".jpg"):
			img = cv2.imread(path + file)
			img_list.append(img)

		if file.endswith(".txt"):
			with open(path + file) as f:
				lines = [line.rstrip() for line in f]
			line_idx = 0
			while (line_idx < len(lines)):
				focal_len = float(lines[line_idx + 11])
				focal_len_list.append(focal_len)
				line_idx += 13

	return img_list, focal_len_list


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, default="parrington", choices=["parrington","grail","denny","csie"])
	args = parser.parse_args()
	path = "../data/" + args.dataset + "/"

	print('reading data...\n')
	img_list, focal_len_list = read_data(path)
	
	print('doing cylindrical projection...\n')
	for idx, img in enumerate(img_list):
		img = cylindrical_proj(img, focal_len_list[idx])

	print('doing keypoints detection and calculating SIFT descriptor...\n')
	sift = SIFT_descriptor()
	keypoints_list = []
	des_list = []
	for i, img in enumerate(img_list):
		threshold_ratio = 0.01
		sigma = 0.5
		t = 0.2
		if "parrington" in path:
			sigma = 1 if i not in [4,5,7] else 0.75
			sigma = sigma if i not in [9] else 0.5
		if "grail" in path:
			sigma = 1 if i in [6,15] else sigma
			sigma = 2 if i in [10] else sigma
			t = 0.4
		if "csie" in path:
			sigma = 0
			sigma = 0.5 if i in [4] else sigma
			threshold_ratio = 0.1

		keypoints = harris_corner_detection(img, threshold_ratio=threshold_ratio, sigma=sigma)
		keypoints_list.append(keypoints)
		des = sift.cal_descriptor(img, keypoints, threshold=t)
		des_list.append(des)
		
	# magic number for each dataset
	thresh_list = [0.9 for i in range(len(keypoints_list)-1)]
	if "parrington" in path:
		thresh_list = [0.85 for i in range(len(keypoints_list)-1)]
		thresh_list[5] = 0.825
		thresh_list[6] = 0.9
		thresh_list[8] = 0.8
		thresh_list[9] = 0.75
		thresh_list[11] = 0.8
		thresh_list[13] = 0.8

	if "grail" in path:
		thresh_list = [0.8 for i in range(len(keypoints_list)-1)]
		thresh_list[2] = 0.6
		thresh_list[6] = 0.2
		thresh_list[7] = 0.6
		thresh_list[8] = 0.65
		thresh_list[9] = 0.75
		thresh_list[10] = 0.75
		thresh_list[15] = 0.7
		thresh_list[16] = 0.6

	if "csie" in path:
		thresh_list = [0.6 for i in range(len(keypoints_list)-1)]
		thresh_list[1] = 0.4
		thresh_list[3] = 0.48
		thresh_list[4] = 0.55
		thresh_list[5] = 0.5
		thresh_list[6] = 0.48
		thresh_list[6] = 0.5
		thresh_list[7] = 0.5
		thresh_list[9] = 0.5

	np.random.seed(666) # fix seed

	print('find the translation offset to match adjacent images...\n')
	offsets_x = []
	offsets_y = []
	for idx in range(len(keypoints_list)-1):
		# Convert list of numpy coords, to list of cv2 keypoints
		cv_kp1 = []
		for k in keypoints_list[idx]:
			y = float(k[0])
			x = float(k[1])
			cv_kp1.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)
		cv_kp2 = []
		for k in keypoints_list[idx+1]:
			y = float(k[0])
			x = float(k[1])
			cv_kp2.append(cv2.KeyPoint(x, y, size=1, angle=0, response=1, octave=1, class_id=0)) # (x, y, size, angle, response, octave, class_id)
		
		matches2 = filter_matches(des_list[idx], des_list[idx+1], cv_kp1, cv_kp2, thresh_list[idx]) # [x1, y1, x2, y2]

		blursigma = 1
		y_offset_lim = 24
		wind_sz = 400
		x_offset_lim = 2
		if "grail" in path:
			blursigma = 1
			y_offset_lim = 10
			wind_sz = 2500
			x_offset_lim = 2.25
		if "csie" in path:
			y_offset_lim = 75
			wind_sz = 2500
			x_offset_lim = 1.75

		off_x, off_y = ransac_img_matching(matches2, img_list[idx], img_list[idx+1], wind_sz=wind_sz, blursigma=blursigma, x_offset_lim=x_offset_lim, y_offset_lim=y_offset_lim)
		offsets_x.append(off_x)
		offsets_y.append(off_y)
		print(f"idx: {idx}")
		print(f'final offset: {off_x}, {off_y}\n')
		'''
		# for visually debugging, please don't delete this:
		draw_matches(img_list[idx], img_list[idx+1], matches2)
		draw_stitch_result(off_x, off_y, img_list[idx], img_list[idx+1])
		'''


	print('stitching panorama...\n')
	blend = True
	if "csie" in path:
		blend = False
	panorama = Panorama(img_list, offsets_x, offsets_y, blend)
	panorama.cal_panorama()
	result = panorama.__getpanorama__().astype(np.uint8)
	# plt.imshow(result)
	# plt.show()
	cv2.imwrite("./../data/output/panorama.jpg", result)
	
	

	