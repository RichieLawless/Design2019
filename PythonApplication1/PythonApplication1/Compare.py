import cv2
import scipy
import skimage
import numpy as np
from skimage import io
from skimage import data, img_as_float
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import argparse
import imutils
import glob

print ("Below are the current saved png files to choose from \n")

print (glob.glob('./*.png'))

selectimg1 = input('Please input the name of the first picture you would like to compare, be sure to include .png: \n')
selectimg2 = input('Please enter the target picture you want to compare against, be sure to include .png \n')


img1 = io.imread(selectimg1)
img2 = io.imread(selectimg2)


######
#Version 1.0
######
#if img1.shape == img2.shape:
#    print("The images have same size and channels")
 #   difference = cv2.subtract(img1, img2)
  #  b, g, r = cv2.split(difference)
 #
 #
#if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
 #   print("The images are completely Equal")
#else:
#    print("The images are not the same")

#####
#Version 2.0
#####
def mse(img1, img2):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
	err /= float(img1.shape[0] * img2.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(img1, img2, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(img1, img2)
	s = ssim(img1, img2)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(img1, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(img2, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()
#load the images -- the original, the original + target,
# and the original + side
original = img1
target = img2

 
# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)


# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("target", target)
 
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 
# show the figure
plt.show()
 
# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, target, "Original vs. target")

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

