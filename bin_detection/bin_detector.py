'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os, cv2
# from roipoly import RoiPoly
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from numpy import save, load
import yaml

img_number = 61

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.training_data_blue = []
		self.training_data_not_blue = []
		self.parameters = np.zeros((3, 3)).tolist()
		self.img_number = 61
		'''
			Initializer for the training set
			This part is commented so we don't train it on gradescope
			Also saving the model makes life easier.
		'''
		# self.initialize_training()
		
		'''
			Let's load the model from our folder
		
		'''
		self.load_model()
		# print(self.parameters)
		'''
		
		
		This will be the mean and variance of data--set
		I made in this format so that matrix calculation becomes easier.
		So our training time is as little as possible
		'''
		"""
		-----------------|-----------------|-------------------|---------------|
		                 |    not blue     |     not blue      |      blue     |
		-----------------|-----------------|-------------------|---------------|
		    probability  |                 |                   |               |
		-----------------|-----------------|-------------------|---------------|
		       mean      |   (r, g, b)     |     (r, g, b)     |   (r, g, b)   |
		-----------------|-----------------|-------------------|---------------|
		     variance    |   (r, g, b)     |     (r, g, b)     |   (r, g, b)   |
		-----------------|-----------------|-------------------|---------------|
		"""
		
		
		pass

	'''
		This method is used to initialize the training for the images
	'''
	def initialize_training(self):
		folder = 'data/training'
		# print(folder)
		# flag = 0
		for filename in os.listdir(folder):
			print(filename)
			img = cv2.imread(os.path.join(folder,filename))
			self.make_blue_training_data(img)
			self.make_not_blue_training_data(img)
			
			# flag = flag + 1
			# if flag == 1:
				# break
				
				
		self.training_data_blue = np.asarray(self.training_data_blue)
		self.training_data_not_blue = np.asarray(self.training_data_not_blue)
		self.build_model(0)
		self.build_model(1)
		
		save('data.npy', self.parameters)
		a = load('data.npy', allow_pickle=True)
		print(a == self.parameters)
		
		return
        
	'''
		We apply roiploy to get blue training data
		We store the training data to out custom list
		The data keeps on adding as we see new images
	'''
	def make_blue_training_data(self, img):
		'''
			The mask that we obtained from roipoly
			is used to create the dataset for our interest blue and non-blue class
			we will create a new data-set training data
			Where training_data [0] will be related to blue
			And training_data[1] will be related to non-blue class
		'''
		fig, ax = plt.subplots()
		dim = (512, 512)
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		ax.imshow(img)     
		my_roi = RoiPoly(fig=fig, ax=ax, color='r')
		mask = my_roi.get_mask(img)
		mask = mask.astype(np.uint32)
		
		blue = self.training_data_blue
		
		row, col = mask.shape
		mask = mask.tolist()
		
		img = img.tolist()
		for i in range(row):
			temp = []
			for j in range(col):
				if mask[i][j] == 1:
					blue.append(img[i][j])
		
		return


	'''
		We apply roiploy to get not blue training data
		We store the training data to out custom list
		The data keeps on adding as we see new images
	'''
	def make_not_blue_training_data(self, img):
		'''
			The mask that we obtained from roipoly
			is used to create the dataset for our interest blue and non-blue class
			we will create a new data-set training data
			Where training_data [0] will be related to blue
			And training_data[1] will be related to non-blue class
		'''
		fig, ax = plt.subplots()
		dim = (512, 512)
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		ax.imshow(img)     
		my_roi = RoiPoly(fig=fig, ax=ax, color='r')
		mask = my_roi.get_mask(img)
		mask = mask.astype(np.uint32)
		
		not_blue = self.training_data_not_blue
		
		row, col = mask.shape
		mask = mask.tolist()
		
		img = img.tolist()
		for i in range(row):
			temp = []
			for j in range(col):
				if mask[i][j] == 0:
					not_blue.append(img[i][j])
		
		return

	'''
		Once we have our blue and non-blue data
		We training the model and get the parameters
		Store them in parameters list
	'''
	def build_model(self, label):
		'''
			This method is used to build the model
			The parameters for this model will be stored in the parameter matrix
		'''
		total_points_blue = self.training_data_blue.shape[0]
		total_points_not_blue = self.training_data_not_blue.shape[0]
		
		if label == 0:
			# We are going to the train the parameters for blue class
			probability = total_points_blue / ( total_points_blue + total_points_not_blue)
			mean = np.mean(self.training_data_blue, axis=0).tolist()
			variance = np.var(self.training_data_blue, axis=0).tolist()
			self.parameters[0][2] = probability
			self.parameters[1][2] = mean
			self.parameters[2][2] = variance
		
		if label == 1:
			# We are going to the train the parameters for blue class
			probability = total_points_not_blue / ( total_points_blue + total_points_not_blue)
			mean = np.mean(self.training_data_not_blue, axis=0).tolist()
			variance = np.var(self.training_data_not_blue, axis=0).tolist()
			self.parameters[0][0] = probability
			self.parameters[1][0] = mean
			self.parameters[2][0] = variance
			
			self.parameters[0][1] = probability
			self.parameters[1][1] = mean
			self.parameters[2][1] = variance
		return


	'''
		This method is used to load the model to our python script
	'''
	def load_model(self):
		folder_path = os.path.dirname(os.path.abspath(__file__))
		model_params_file = os.path.join(folder_path, 'data.npy')
		self.parameters = load(model_params_file, allow_pickle=True)
		
		return


	'''
		This method is just the implementation of gamma correction
		My trained model works better on darker shades of blue
		For this reason, I make my image slilghtly darker to get better results
	'''
	def gammaCorrection(self, src, gamma):
		invGamma = 1 / gamma
		table = [((i / 255) ** invGamma) * 255 for i in range(256)]
		table = np.array(table, np.uint8)
		return cv2.LUT(src, table)   


	'''
		This method is used to segment tha image
		Here we use our trained model to classify that whether a given pixel is blue or not blue.
		Once we classify that, we convert the image with black and white image
		Here white is for blue bin and rest is black
		Return the segmented image
	'''
	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach
		row, col, colors  = img.shape
		# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		gammaImg = self.gammaCorrection(img, 1.1)
		# cv2.imshow("Gamma", gammaImg)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
		X = gammaImg.reshape(row*col, colors)
		
		label_y = np.zeros((row*col, colors))
		for color in range(colors):
			probability = self.parameters[0][color]
			mean = self.parameters[1][color]
			variance = self.parameters[2][color]
			
			part_a = np.log(probability)
			part_b = np.zeros((row*col))
			for dim in range(3):
				part_b = np.add(part_b , (-1/2)*( np.log(variance[dim]) ) + (-1/2)*(pow(X[:, dim] - mean[dim], 2)/variance[dim]))
				
			label_y[:, color] = part_b + part_a
			
		y = np.argmax(label_y, axis=1) + 1
		y = y.reshape(row, col)
		y = y.tolist()
		
		mask_img = np.zeros((row, col)).tolist()
		
		for data_i in range(row):
			for data_j in range(col):
				if y[data_i][data_j] == 3:
					mask_img[data_i][data_j] = [255, 255, 255]
				else:
					mask_img[data_i][data_j] = [0, 0, 0]
				
		mask_img = np.array(mask_img, dtype=np.uint8)
		
		# cv2.imshow("Mask Image", mask_img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img
    

	'''
		This method is useful in finding the bounding box of the images
		Here the input is the masked image we get from the previous method (segment_image)
		We find the approriate list of bounding box around our bins.
		And return those list
	'''
	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		row, col, colors  = img.shape
		b_row, b_col = row*0.01, col*0.01
		
		imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		'''
			All the morphological operation in order to get the accuracy better.
			Did dilation for the first time so that important feature which are scattered are retained
			Did erosion so that the small masks are removed.
			Did dilation for the second time so that fill in blank spaces at the edges.
		'''
		kernel_1 = np.ones((4, 4), np.uint8)
		kernel_2 = np.ones((6, 6), np.uint8)
		kernel_3 = np.ones((8, 8), np.uint8)
		img_dilation = cv2.dilate(imgray, kernel_1, iterations=4)
		img_erosion = cv2.erode(img_dilation, kernel_2, iterations=10)
		img_dilation = cv2.dilate(img_erosion, kernel_3, iterations=6)
		ret, thresh = cv2.threshold(img_dilation, 127, 255, 0)
		
		'''
			Here we find the contours in our images used the cv2.findcontours
			It gives also gives the hierachy of the contours but our hierachy is finding rectangle
			I check the ratio of the boc should be greater than 1% of the image
			And measurement of the width is less than length
			Just this was helping me with good accuracy.
		'''
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		boxes = []
		# print()
		for cnt in contours:
			x,y,w,h = bbox = cv2.boundingRect(cnt)
			# if h>b_row and w>b_col:
			if h > w:
				# print("Height: ", h, "\tWidth: ", w)
				boxes.append([x, y, x+w, y+h])
				img_dilation = cv2.rectangle(img, (x,y),(x+w,y+h),(255, 0, 0 ), 4)
		
		folder = 'data/output/'
		folder = folder + str(self.img_number) + ".png"
		print(folder)
		cv2.imwrite(folder, img_dilation)
		cv2.imshow("Contour", img_dilation)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		self.img_number = self.img_number + 1
		
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes
    
    
	'''
		This method was used for testing purposes
		Kept it just for my reference
	'''
	def get_mask(self, img):
		fig, ax = plt.subplots()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		ax.imshow(img)     
		my_roi = RoiPoly(fig=fig, ax=ax, color='r')
		mask = my_roi.get_mask(img)
		
		mask = mask.astype(np.uint32)
		self.make_training_data(img, mask)
		
		return


	'''
		This method was used for testing purposes
		Kept it just for my reference
	'''
	def make_training_data(self, img, mask):
		'''
			The mask that we obtained from roipoly
			is used to create the dataset for our interest blue and non-blue class
			we will create a new data-set training data
			Where training_data [0] will be related to blue
			And training_data[1] will be related to non-blue class
		'''
		blue = self.training_data_blue
		not_blue = self.training_data_not_blue
		
		row, col = mask.shape
		mask = mask.tolist()
		
		img = img.tolist()
		
		for i in range(row):
			temp = []
			for j in range(col):
				if mask[i][j] == 1:
					blue.append(img[i][j])
				else:
					not_blue.append(img[i][j])
		
		return