import cv2
from cylindrical_proj import cylindrical_proj
import numpy as np
import matplotlib.pyplot as plt

class Panorama():
	def __init__(self, img_list, offsets_x, offsets_y, blend=True):
		self.img_list = img_list
		self.offsets_x = offsets_x
		self.offsets_y = offsets_y
		self.blend = blend

		self.left_intersect = []
		self.right_intersect = []

	def init_panorma(self):
		H,W,C = self.img_list[0].shape
		H = int(H*2)
		W = int(W*len(self.img_list)*0.8)
		self.panorama = np.zeros((H,W,C))

	def cal_intersects(self):
		for idx in range(len(self.img_list)-1):
			img1 = self.img_list[idx]
			img2 = self.img_list[idx+1]

			x_offset = self.offsets_x[idx]
			y_offset = self.offsets_y[idx]

			x = np.linspace(0, 1, abs(x_offset))
			H = img1.shape[0]
			y = np.ones((1,H-abs(y_offset)))

			x_blend_weight, _ = np.meshgrid(x,y) 

			if y_offset < 0:
				img1_right = img1[-y_offset:, x_offset:].astype(np.float64)
				img2_left = img2[:y_offset, :-x_offset].astype(np.float64)
			elif y_offset > 0:
				img1_right = img1[y_offset:, x_offset:].astype(np.float64)
				img2_left = img2[:-y_offset, :-x_offset].astype(np.float64)
			else:
				img1_right = img1[:, x_offset:].astype(np.float64)
				img2_left = img2[:, :-x_offset].astype(np.float64)

			img2_left[:,:,0] *= x_blend_weight
			img2_left[:,:,1] *= x_blend_weight
			img2_left[:,:,2] *= x_blend_weight

			x_blend_weight = np.flip(x_blend_weight, axis=1)
			img1_right[:,:,0] *= x_blend_weight
			img1_right[:,:,1] *= x_blend_weight
			img1_right[:,:,2] *= x_blend_weight

			self.right_intersect.append(img1_right)
			self.left_intersect.append(img2_left)

	def stitch_panorama(self):
		curr_up = 400
		curr_right = 0
		for idx in range(len(self.img_list)-1):
			img2 = self.img_list[idx+1]
			x_offset = self.offsets_x[idx]
			y_offset = self.offsets_y[idx]

			if idx==0: 
				# stitch img1(left) and img2(right)
				img1 = self.img_list[idx]
				h,w,c = img1.shape
				self.panorama[curr_up:curr_up+h, :w, :] = img1 
				w1 = int(w+x_offset)
				w2 = int(w1+w)
				h,w,c = img2.shape
				self.panorama[curr_up-y_offset:curr_up-y_offset+h, w1:w2 , :] = img2
				if self.blend:
					# blending of intersection area
					h1,w1,_ = self.left_intersect[idx].shape
					self.panorama[curr_up-y_offset:curr_up-y_offset+h1, w-w1:w, :] = self.left_intersect[idx] + self.right_intersect[idx]
			else:
				# stitch img2 only
				h,w,c = img2.shape
				w1 = int(curr_right+x_offset)
				w2 = int(w1+w)
				self.panorama[curr_up-y_offset:curr_up-y_offset+h, w1:w2 , :] = img2
				if self.blend:
					# blending of intersection area
					h1,w1,_ = self.left_intersect[idx].shape
					self.panorama[curr_up-y_offset:curr_up-y_offset+h1, curr_right-w1:curr_right, :] = self.left_intersect[idx] + self.right_intersect[idx]

			# update parameter for the next image
			curr_right = w2
			curr_up -= y_offset
			img1 = img2
	
	def trim_panorama(self):
		row = self.panorama[600, :, :]
		trim_right = 0
		for i in range(row.shape[0]-1, 0, -1):
			if np.sum(row[i, :]) > 0:
				trim_right = i
				break

		half = int(self.panorama.shape[1]/2)
		col = self.panorama[:, half, :]
		trim_up = 0
		for i in range(col.shape[0]):
			if np.sum(col[i, :]) > 0:
				trim_up = i
				break

		col = self.panorama[:, half, :]
		trim_bottom = 0
		for i in range(col.shape[0]-1, 0, -1):
			if np.sum(col[i, :]) > 0:
				trim_bottom = i
				break
		print(f'trim panorama, up: {trim_up}, trim panorama, bot: {trim_bottom}, right: {trim_right}')
		self.panorama = self.panorama[trim_up:trim_bottom, :trim_right, :]


	def cal_panorama(self):
		self.init_panorma()
		self.cal_intersects()
		self.stitch_panorama()
		self.trim_panorama()

	def __getpanorama__(self):
		return self.panorama


if __name__ == '__main__':
	test_img_name1 = "./../data/parrington/prtn07.jpg"
	test_img_name2 = "./../data/parrington/prtn06.jpg"
	test_img_name3 = "./../data/parrington/prtn05.jpg"

	img1 = cv2.imread(test_img_name1)
	img2 = cv2.imread(test_img_name2)
	img3 = cv2.imread(test_img_name3)

	img1 = cylindrical_proj(img1, 705.645)
	img2 = cylindrical_proj(img2, 705.327)
	img3 = cylindrical_proj(img3, 704.696)

	img_list = [img1, img2, img3]

	# # between 05 and 06
	# off_x = -131
	# off_y = -3

	# # between 06 and 07
	# off_x = -141
	# off_y = -5

	# offsets_x = [-131, -141]
	# offsets_y = [-4, -5]
	offsets_x = [-131, -121]
	offsets_y = [-4, -3]

	panorama = Panorama(img_list, offsets_x, offsets_y)

	panorama.init_panorma()
	# a = panorama.test()
	b = panorama.cal_intersects()
	c = panorama.stitch_panorama()
	plt.imshow(c/255)
	plt.show()

	

	


