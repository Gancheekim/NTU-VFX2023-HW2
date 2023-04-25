import numpy as np
	
def cylindrical_proj(img, f):
	warp_img = np.zeros_like(img)
	H,W,_ = img.shape

	# pre-compute the final warpping (x,y) coordinate as LUT
	x = np.linspace(0, W-1, W)
	y = np.linspace(0, H-1, H)
	# the original coordinate
	x_ori_mat, y_ori_mat = np.meshgrid(x,y) 
	# the coordinate after cylindrical projection, which the center must be at (H/2, W/2)
	x_mat, y_mat = x_ori_mat - W/2, y_ori_mat - H/2
	x_mat = f*np.arctan(x_mat/f)
	y_mat = f*y_mat/np.sqrt(x_mat*x_mat + f*f)
	# apply the offset back
	x_mat += W/2
	y_mat += H/2
	# make sure the coordinate matrices are int
	x_mat = x_mat.astype(np.int32)
	y_mat = y_mat.astype(np.int32)
	x_ori_mat = x_ori_mat.astype(np.int32)
	y_ori_mat = y_ori_mat.astype(np.int32)

	# foward warpping
	warp_img[y_mat, x_mat, :] = img[y_ori_mat, x_ori_mat, :]

	return warp_img
	

if __name__ == "__main__":
	import cv2
	import matplotlib.pyplot as plt
	
	test_img_name1 = "./../data/parrington/prtn13.jpg"
	img1 = cv2.imread(test_img_name1)
	warpped = cylindrical_proj(img1, 704.676)
	
	plt.subplot(1,2,1)
	plt.imshow(img1)
	plt.subplot(1,2,2)
	plt.imshow(warpped)
	plt.show()