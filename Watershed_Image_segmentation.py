import numpy as np
import cv2
import  matplotlib.pyplot as plt
image=cv2.imread("new_image.png")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,225,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel=np.ones((3,3),np.uint8)
sure_bg=cv2.dilate(thresh,kernel,iteration=3)
