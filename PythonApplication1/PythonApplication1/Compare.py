import cv2
import scipy
import skimage
from skimage import io
from skimage.measure import compare_ssim
import argparse
import imutils
import picture

img1 = io.imread("img1.png")
img2 = io.imread("img2.png")


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