import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import requests

image=cv2.imread("new_image.png",cv2.IMREAD_GRAYSCALE)
edges=cv2.Canny(image,50,150)
cv2.imshow("image",edges)
blur=cv2.GaussianBlur(image,(5,5),0)
plt.figure(figsize=(8,6))

#plt.subplot(1,2,1),plt.imshow(image),plt.title('Original')
#plt.subplot(1,2,2),plt.imshow(edges),plt.title("edges")
blur_color_image=cv2.applyColorMap(blur,cv2.IMREAD_COLOR_BGR)
cv2.imread('Colored_blur_image',blur_color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
