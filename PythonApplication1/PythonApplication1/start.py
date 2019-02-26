import cv2
import os
import sys
import subprocess
import tkinter
print ("starting demo script one")
confirm = input("Would you like to start the demo? Y/N \n")


if (confirm.lower() == "y"):
    print("Press c to confirm picture \n")
    os.system("python picture.py")
    
    input('Press enter to continue to the next picture: ')
    os.system("python picture.py")
    

    print("We will now move on to the compare function")
    os.system("python Compare.py")
    input('Press any key to end the demo')

else:
    os._exit(1)