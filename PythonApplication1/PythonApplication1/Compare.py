import cv2
import scipy
import skimage
import numpy as np
from skimage import io
from skimage import data, img_as_float
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.transform import resize
from scipy.ndimage import imread
import argparse
import imutils
import glob
import warnings

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

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier('C:\\Users\\Richie\\Desktop\\External Libraries\\OpenCV\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Richie\\Desktop\\External Libraries\\OpenCV\\etc\\haarcascades\\haarcascades_eye.xml')
'''
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

#### Removed for version 3.0, uses a similar command
# convert the images to grayscale
'''
#original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

original = img1
target = img2
#######
#Version 3.0 addon
#######

# Load in color image for face detection
#original = cv2.imread(original)
#target = cv2.imread(target)

# Convert the image to RGB colorspace
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
# Make a copy of the original image to draw face detections on
image_original = np.copy(original)
image_target = np.copy(original)

# Convert the image to gray 
gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
# Detect faces in the image using pre-trained face dectector
faces_original = face_cascade.detectMultiScale(gray_original, 1.25, 6)
faces_target = face_cascade.detectMultiScale(gray_target, 1.25, 6)


# Get bounding box for each face in original
for f in faces_original:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_original, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop_original = gray_original[y:y+h, x:x+w]
    cv2.imwrite("C:\\Users\\Richie\\source\\repos\\Design2019\\PythonApplication1\\PythonApplication1\\FCO.png", face_crop_original)
#Get bounding box for each face in target


for t in faces_target:
    x, y, w, h = [ v for v in t ]
    cv2.rectangle(image_target, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop_target = gray_target[y:y+h, x:x+w]
    cv2.imwrite("C:\\Users\\Richie\\source\\repos\\Design2019\\PythonApplication1\\PythonApplication1\\FCT.png", face_crop_target)
# Display the image with the bounding boxes
fig = plt.figure(figsize = (9,9))
axl = fig.add_subplot(111)
axl.set_xticks([])
axl.set_yticks([])

axl.set_title("Face detection")
axl.imshow(image_original)
axl.imshow(image_target)

# Display the face crops
fig = plt.figure(figsize = (9,9))
axl = fig.add_subplot(111)
axl.set_xticks([])
axl.set_yticks([])

axl.set_title("Face crops")
axl.imshow(face_crop_original)
axl.imshow(face_crop_target)
face_crop_original = []
face_crop_target = []

for f in faces_original:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_original, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop_original.append(gray_original [y:y+h, x:x+w])

for f in faces_target:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_target, (x,y), (x+w, y+h), (255,0,0), 3)
    face_crop_target.append(gray_target [y:y+h, x:x+w])

for face in face_crop_original:
    cv2.imshow('face',face)
    cv2.waitKey(0)

for face in face_crop_target:
    cv2.imshow('face',face)
    cv2.waitKey(0)

#cv2.imwrite(str(face_crop_original)+'.png', frame)
#cv2.imwrite(str(face_crop_target)+'.png', frame)




#######
#End of version 3.0
#######



#def mse(face_crop_original, face_crop_target):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	#err = np.sum((face_crop_original.astype("float") - face_crop_target.astype("float")) ** 2)
	#err /= float(face_crop_original.shape[0] * face_crop_target.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	#return err
 
def compare_images(face_crop_original, face_crop_target, multichannel = True):  #might need to change full=true to title
	# compute the mean squared error and structural similarity
	# index for the images
	#m = mse(face_crop_original, face_crop_target)
    face_crop_original = np.array([[],[],[]]) 
    face_crop_target = np.array([[],[],[]])
    s = ssim(face_crop_original, face_crop_target, multichannel=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
	# setup the figure
    #fig = plt.figure(title)
	#plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(face_crop_original, cmap = plt.cm.gray)    
    plt.axis("off")
 
	# show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(face_crop_target, cmap = plt.cm.gray)
    plt.axis("off")
 
	# show the images
    plt.show()
#load the images -- the original, the original + target,
# and the original + side
#original = img1
#target = img2

fig = plt.figure("Images")
images = ("Original", face_crop_original), ("target", face_crop_target)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	#plt.imshow(image, cmap = plt.cm.gray)
	#plt.axis("off")
 
# show the figure
#plt.show()
 
# compare the images
height = 2**10
width = 2**10

#gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
#gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)


ssim(face_crop_original, face_crop_target, multichannel=True, win_size=3)
#ssim(multichannel = True)

compare_images(face_crop_original, face_crop_target, "Original vs. target")

cv2.waitKey(0)
cv2.destroyAllWindows()

#### Removed for version 3.0, uses a similar command
# convert the images to grayscale
#original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
