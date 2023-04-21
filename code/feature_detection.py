import cv2
import numpy as np
import matplotlib.pyplot as plt 
from skimage.feature import peak_local_max

def harris_corner_detection(img, gauss_filt_size=3, k=0.04, threshold_ratio=0.01):
	'''
	:img: input image
	:gauss_filt_size: kernel size for the gaussian blur
	:k: constant used to compute the response matrix R
	:threshold_ratio: minimum threshold level to be treated as a feature point
	----------
	output: a N-by-2 array, representing N feature points of (y,x) coordinate
	'''
	# 0. only uses grayscale image for computing gradient
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# 1. compute x and y derivatives of image
	blurred = cv2.GaussianBlur(img, (gauss_filt_size,gauss_filt_size), 0)
	Ix, Iy = np.gradient(blurred)

	# 2. compute products of derivatives at every pixel
	Ix2 = Ix * Ix
	Iy2 = Iy * Iy
	Ixy = Ix * Iy

	# 3. compute the sums of the products of derivatives at each pixel
	Sx2 = cv2.GaussianBlur(Ix2, (gauss_filt_size,gauss_filt_size), 0)
	Sy2 = cv2.GaussianBlur(Iy2, (gauss_filt_size,gauss_filt_size), 0)
	Sxy = cv2.GaussianBlur(Ixy, (gauss_filt_size,gauss_filt_size), 0)

	# 4. Define the matrix at each pixel
	# M(x,y) = [Sx2, Sxy]
	#          [Sxy, Sy2]
	# but we dont have to really compute this

	# 5. Compute the response of the detector at each pixel
	detM = Sx2*Sy2 - Sxy*Sxy
	traceM = Sx2*Sy2
	R = detM - k*(traceM*traceM)

	# 6. Threshold on value of R; compute nonmax suppression.
	threshold = np.max(R) * threshold_ratio
	R[R < threshold] = 0
	feature_pts = peak_local_max(R, min_distance=10) 
	# dim: (N,2), each representing [y, x]
	return feature_pts 


if __name__ == "__main__":
	test_img_name = "./../data/test/chessboard.jpg"
	img = cv2.imread(test_img_name)

	feature_pts = harris_corner_detection(img)
	plt.imshow(img)
	plt.plot(feature_pts[:, 1], feature_pts[:, 0], 'r.')
	plt.show()
