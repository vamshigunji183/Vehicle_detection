import numpy as np
import cv2
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from settings import *


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	"""
	Return HOG features and visualization (optionally)
	"""
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=True, feature_vector=False)
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=False, feature_vector=feature_vec)
		return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel()
	# Return the feature vector
	return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for image in imgs:
		image_features = []
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
			elif color_space == 'GRAY':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				feature_image = np.stack((feature_image, feature_image, feature_image), axis=2)  # keep shape
		else: feature_image = np.copy(image)

		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			image_features.append(spatial_features)
		if hist_feat == True:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			image_features.append(hist_features)
		if hog_feat == True:
		# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel],
										orient, pix_per_cell, cell_per_block,
										vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
							pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			image_features.append(hog_features)
		features.append(np.concatenate(image_features))
	# Return list of feature vectors
	return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
	#1) Define an empty list to receive features
	img_features = []
	#2) Apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		elif color_space == 'GRAY':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				feature_image = np.stack((feature_image, feature_image, feature_image), axis=2)  # keep shape
	else: feature_image = np.copy(img)
	#3) Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		#4) Append features to list
		img_features.append(spatial_features)
	#5) Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		#6) Append features to list
		img_features.append(hist_features)
	#7) Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel],
									orient, pix_per_cell, cell_per_block,
									vis=False, feature_vec=True))
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		#8) Append features to list
		img_features.append(hog_features)

	#9) Return concatenated array of features
	return np.concatenate(img_features)


if __name__ == '__main__':

	# Vehicle image
	image = mpimg.imread('example_images/vehicle.png')
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Call our function with vis=True to see an image output
	features, hog_image = get_hog_features(gray, orient,
		pix_per_cell, cell_per_block,
		vis=True, feature_vec=False)

	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(image)
	plt.title('Example Car Image')
	plt.subplot(122)
	plt.imshow(hog_image, cmap='gray')
	plt.title('HOG Visualization')
	plt.show()

	# Non-car image
	image = mpimg.imread('example_images/non_vehicle.png')
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Call our function with vis=True to see an image output
	features, hog_image = get_hog_features(gray, orient,
		pix_per_cell, cell_per_block,
		vis=True, feature_vec=False)

	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(image)
	plt.title('Example Non-Car Image')
	plt.subplot(122)
	plt.imshow(hog_image, cmap='gray')
	plt.title('HOG Visualization')
	plt.show()
