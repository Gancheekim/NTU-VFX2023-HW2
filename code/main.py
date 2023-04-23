from cylindrical_proj import cylindrical_proj
from feature_detection import harris_corner_detection
from feature_descriptor import SIFT_descriptor

import argparse
import cv2
import os
import matplotlib.pyplot as plt


def data_process(path):
	img_list = []
	focal_len_list = []
	for file in os.listdir(path):
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
	path = "../data/parrington/"
	img_list, focal_len_list = data_process(path)
	
	for idx, img in enumerate(img_list):
		img = cylindrical_proj(img, focal_len_list[idx])

	

	