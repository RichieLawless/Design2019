import cv2
import scipy
import skimage
from skimage import io
from skimage.measure import compare_ssim
import argparse
import imutils
import glob

print ("below are the current saved png files to choose from \n")

print (glob.glob('./*.png'))

selectimg1 = input('Please input the name of the first picture you would like to compare, be sure to include .png: \n')
selectimg2 = input('Please input the name of the second picture you would like to compare, be sure to include .png: \n')


img1 = io.imread(selectimg1)
img2 = io.imread(selectimg2)


if img1.shape == img2.shape:
    print("The images have same size and channels")
    difference = cv2.subtract(img1, img2)
    b, g, r = cv2.split(difference)
 
 
if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("The images are completely Equal")
else:
    print("The images are not the same")

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
def get_histogram(img): 
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.

  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w) 


def normalize_exposure(img):

  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)
'''