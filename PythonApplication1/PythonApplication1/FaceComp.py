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
from scipy.ndimage import uniform_filter, gaussian_filter
from PIL import Image

face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

print ("Below are the current saved png files to choose from \n")

print (glob.glob('./*.png'))

selectimg1 = input('Please input the name of the first picture you would like to compare, be sure to include .png: \n')
selectimg2 = input('Please enter the target picture you want to compare against, be sure to include .png \n')


img1 = io.imread(selectimg1)
img2 = io.imread(selectimg2)

original = img1
target = img2

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



fig = plt.figure("Images")
images = ("Original", img1), ("target", img2)
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

for f in faces_original:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_original, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop_original = gray_original[y:y+h, x:x+w]
#Get bounding box for each face in target


for t in faces_target:
    x, y, w, h = [ v for v in t ]
    cv2.rectangle(image_target, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop_target = gray_target[y:y+h, x:x+w]



def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))
 
    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print('The resized image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))
    resized_image.save(output_image_path)

def mse(img1, img2):

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
	#plt.suptitle("SSIM: %.2f" % (s))
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

# Display the image with the bounding boxes
fig = plt.figure(figsize = (9,9))
axl = fig.add_subplot(111)
axl.set_xticks
axl.set_yticks([])

axl.set_title("Face detection")
axl.imshow(image_original)
axl.imshow(image_target)

cv2.imwrite('face_crop_original.png', face_crop_original)
cv2.imwrite('face_crop_target.png', face_crop_target)


resize_image(input_image_path='face_crop_original.png',
              output_image_path='face_crop_original.png',
              size=(1920, 1080))
resize_image(input_image_path='face_crop_target.png',
              output_image_path='face_crop_target.png',
              size=(1920, 1080))

original = io.imread('face_crop_original.png')
target = io.imread('face_crop_target.png')

compare_images(original, target, "Original vs. target")

cv2.waitKey(0)
cv2.destroyAllWindows()