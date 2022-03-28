'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

# from generate_rgb_data import read_pixels
import numpy as np
from numpy import save, load
import math
import os

class PixelClassifier():
	def __init__(self):
		'''
			Initilize your classifier with any parameters and attributes you need
		'''
		# Inititalizing the data-set with enpty list
		# Once we run to each folder of the training data and fill up this list
		self.X = []
		self.folder = ""
		self.label = 0
		self.length = 0
		self.total_length = 0
		self.parameters = [ [ 0 for i in range(3) ] for j in range(3) ]  
		
		# This will be the mean and variance of dataset
		"""
		-----------------|----------------|---------------|---------------|
		                 |       red      |     green     |     blue      |
		-----------------|----------------|---------------|---------------|
		    probability  |                |               |               | 
		-----------------|----------------|---------------|---------------|
		       mean      |   (r, g, b)    |   (r, g, b)   |  (r, g, b)    |
		-----------------|----------------|---------------|---------------|
		     variance    |   (r, g, b)    |   (r, g, b)   |  (r, g, b)    |
		-----------------|----------------|---------------|---------------|   
		"""
		
            # self.train_model()
		self.load_model()
		# print(self.parameters)
		pass


	# This method is used to create the dataset
	def train_model(self):
		# Training folder with the labels red
		self.folder = 'data/training/red'
		self.label = 1
		self.length = 0
		self.create_dataset()
		self.red_data_length = self.length
		
		# Training folder with the labels green
		self.folder = 'data/training/green'
		self.label = 2
		self.length = 0
		self.create_dataset()
		self.green_data_length = self.length
		
		# Training folder with the labels blue
		self.folder = 'data/training/blue'
		self.label = 3
		self.length = 0
		self.create_dataset()
		self.blue_data_length = self.length 
		
		
		# Once the data-set is created 
		# The data-set is stored in X
		# Here will need some more intializers
		# These initializer are the parameters of our model
		
		# This is the prior probabilites of our model
		self.total_length = self.red_data_length + self.blue_data_length + self.green_data_length
		# print(self.total_length)
		pass

	# This method is used to create the dataset
	def create_dataset(self):
		'''
			Create a dataset
			Here the table will be like [R, G, B, Y]
			R, G and B represents the data RGB value for the particular value
			Here the given function "read_pixels" in generate_rgb_data.py will be very helpful
			And Y represnts the correct label. Our labels are {Red, Green, Blue}        
		'''
		# Read the previous data
		previous_data = self.X
		
		# Get data from the current folder location
		# Convert the numpy data to list for easy appending
		data = read_pixels(self.folder)
		data = data.tolist()
		
		# Add the label at the last column
		# At the same time append the data to the previous data
		# Thus we reduce one more for loop
		self.length = len(data)
		for i in range(self.length):
			data[i].append(self.label)
			previous_data.append(data[i])
		
		# Make the X as the previous data
		# Thus X contains all the previous as well as current data
		self.X = previous_data
		pass


	# This method is used to build our naive based model
	def build_model(self):
		'''
			Create a model to train our dataset!    
		'''
		# Now our dataset is present in X because of the "create_dataset" model
		# We know our dataset has 3 labels
		total_label = 3
		for label in range(total_label):
			# Here we have to check those whose label are equal to "label"
			# We find the mean for this for loop
			# We will need another loop just 
			# Because we need to use mean we find here in this loop
			
			# Now we iterate over each data set
			
			# Here total means the to count how many data relates to particular label        
			total = 0
			mean = [0, 0, 0]
			for points in self.X:
				if (points[3]-1) == label:
					mean[0] = mean[0] + points[0]
					mean[1] = mean[1] + points[1]
					mean[2] = mean[2] + points[2]
					total = total + 1
			
			mean[0] = mean[0] / total
			mean[1] = mean[1] / total
			mean[2] = mean[2] / total
			self.parameters[0][label] = total / self.total_length
			self.parameters[1][label] = mean
			
			# Here the second loop is for variance
			variance = [0, 0, 0]
			for points in self.X:
				if (points[3]-1) == label:
					variance[0] = variance[0] + ((points[0] - mean[0])**2)
					variance[1] = variance[1] + ((points[1] - mean[1])**2)
					variance[2] = variance[2] + ((points[2] - mean[2])**2)
			
			variance[0] = variance[0] / total
			variance[1] = variance[1] / total
			variance[2] = variance[2] / total
			self.parameters[2][label] = variance
    
		self.parameters = np.array(self.parameters)
		save('data_pixel.npy', self.parameters)
		pass

    
	def load_model(self):
		folder_path = os.path.dirname(os.path.abspath(__file__))
		model_params_file = os.path.join(folder_path, 'data_pixel.npy')
		self.parameters = load(model_params_file, allow_pickle=True)
		self.parameters = self.parameters.tolist()
		
		return


	
	def classify(self,X):
		'''
			    Classify a set of pixels into red, green, or blue
			    
			    Inputs:
			      X: n x 3 matrix of RGB values
			    Outputs:
			      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
		'''
	################################################################
	# YOUR CODE AFTER THIS LINE
		
		# Just a random classifier for now
		# Replace this with your own approach 
		
		# Once we have our trained model with parameters
		# We can now go ahead and use that to check new data
		# Now we iterate over the new samples that are in X
		row, col = X.shape
		
		label_y = np.zeros((row, col))
		for color in range(col):
			probability = self.parameters[0][color]
			mean = self.parameters[1][color]
			variance = self.parameters[2][color]
			
			part_a = np.log(probability)
			part_b = np.zeros((row))
			for dim in range(3):
				part_b = np.add(part_b , (-1/2)*( np.log(variance[dim]) ) + (-1/2)*(pow(X[:, dim] - mean[dim], 2)/variance[dim]))
			
			label_y[:, color] = part_b + part_a
			
		y = np.argmax(label_y, axis=1) + 1    
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return y

