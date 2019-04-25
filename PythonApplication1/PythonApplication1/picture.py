import numpy as np
import cv2
import os
import tkinter
import tkinter as tk
from tkinter import *
from functools import partial
from tkinter import messagebox
import msvcrt as m
top = tkinter.Tk()
top.geometry('500x500')


def main():
    L1 = Label(top, text="Input name of picture!")
    L1.pack()
    E1 = Entry(top, bd =5)
    E1.pack()
    input = E1.get()


    B = tkinter.Button(top, text='Press here to take a picture!', command = TP(input))
    B.pack()  
    E2 = Entry(top, bd = 5)
    E2.pack
    inputDel = E2.get()
    

    L2 = callable(top, text='Enter the name of the file you want to delete.', command = Delete(inputDel))
    L2.pack
    C = tkinter.Button(top, text = 'Press here to compare two images', command = FaceCompCall)

def TP(args):
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera_port = 0
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imshow('img',frame)
            cv2.imwrite(input+'.png')
            id = 1
            while os.path.exists(input+'.png'):
                id += 1
                camera.release()
                cv2.imwrite(input+str(id)+'.png')
                break

            cv2.destroyAllWindows()
        break
    

def Delete(args):
    os.remove(str(args)+'.png')


def FaceCompCall():
    exec(open("./FaceComp").read())

main()

top.mainloop()


        #camera = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
        #ret,frame = camera.read()
                #while os.path.exists('img'+str(id)+".png"):
         #   cv2.imwrite('img1'+str(id)+'.png', frame)