import tkinter
import tkinter as tk
from tkinter import *
import os
import cv2
import sys

def write_test():
    print("Test")
def exec_label(label):
  execP = 0
def callback():
 os.system("python start.py")

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text = "Take a picture", fg = "blue", command = callback )
button.pack(side = tk.LEFT)

root.mainloop()