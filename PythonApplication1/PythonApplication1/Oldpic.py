import numpy as np
import cv2
import os


camera_port = 0
camera = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
ret,frame = camera.read() # return a single frame in variable `frame`
cap = cv2.VideoCapture(0) 
print('Press c when you are ready to take the picture')
while(True):
    ret, frame = cap.read()
    
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


'''
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''