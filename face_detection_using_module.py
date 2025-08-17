import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread("new_image.png")
blurred=cv2.GaussianBlur(image,(5,5),1)
plt.imshow(blurred)
plt.show()






