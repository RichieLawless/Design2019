import cv2
import os
import sys

print ("starting demo script one")
confirm = input("Would you like to start the demo? Y/N \n")
if (confirm == "y","Y"):
    os.system("python picture.py")
    input('Press enter to continue to the next picture: ')
    os.system("python picture.py")
    input('Press enter to continue to compare the two images?: ')

    os.system("python Compare.py")
else:
    sys.exit(1)