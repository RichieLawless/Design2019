import numpy as np
import cv2
import os


camera_port = 0
camera = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
ret,frame = camera.read() # return a single frame in variable `frame`

while(True):
    cv2.imshow('img1',frame) #display the captured image

    if cv2.waitKey(1) & 0xFF == ord('c'): #save on pressing 'c' 
        cv2.destroyAllWindows()
       # id = 1
       
        #while os.path.exists("img"+str(id)+".png"):
           # id += 1
        chgname = input("Please input your name, this will become the file name: ")
        cv2.imwrite(str(chgname)+'.png', frame)
        #cv2.imwrite('img'+str(id)+'.png',frame)

        camera.release()
        break
        cv2.destroyAllWindows()
        sys.exit(0)
#This captures a single frame or a picture

