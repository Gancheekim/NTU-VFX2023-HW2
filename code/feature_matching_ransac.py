import cv2
import numpy as np
import matplotlib.pyplot as plt 
from feature_detection import harris_corner_detection
from feature_descriptor import SIFT_descriptor


#=====================================================
# Attempt - Ransac Matching
#=====================================================
test_img_name1 = "./../data/parrington/prtn01.jpg"
test_img_name2 = "./../data/parrington/prtn02.jpg"
# test_img_name1 = "./../data/denny/denny00.jpg"
# test_img_name2 = "./../data/denny/denny01.jpg"
img1 = cv2.imread(test_img_name1)
img2 = cv2.imread(test_img_name2)

keypoints1 = harris_corner_detection(img1) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint
keypoints2 = harris_corner_detection(img2) # numpy array of size N-by-2, specifying the corresponding y and x coordinate of a keypoint

sift = SIFT_descriptor()
des1 = sift.cal_descriptor(img1, keypoints1)
des2 = sift.cal_descriptor(img2, keypoints2)

matches = sift.find_matches(des1, des2)

cv_kp1, cv_kp2 = sift.get_cv_keypoints(img1, keypoints1, img2, keypoints2, matches)  # test: get cv keypoints
# Analyze results:
print(f"keypoints1:\n{len(keypoints1)}")                        # prtn01.jpg: found 314 keypoints
print("==================================================\n")
print(f"keypoints2:\n{len(keypoints2)}")                        # prtn02.jpg: found 305 keypoints
print("==================================================\n")
print(f"cv_kp1:\n{len(cv_kp1)}")                                # prtn01.jpg: get 314 cv keypoints
print("==================================================\n")
print(f"cv_kp2:\n{len(cv_kp2)}")                                # prtn02.jpg: get 305 cv keypoints
print("==================================================\n")
print(f"matches:\n{len(matches)}")                              # from two images: 57 matches
