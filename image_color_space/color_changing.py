from PIL import Image
import cv2
weight,height=100,100
image=cv2.imread("new_image.png")
image_BGR=cv2.imread("new_image.png",cv2.IMREAD_COLOR_BGR)
image_RGB=Image.new("RGB",(weight,height),color="white")
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.subplot(1,2,1),plt.imshow(image)
plt.subplot(1,2,2),plt.imshow(image_BGR)
plt.subplot(1,2,3),plt.imshow(image_RGB)
plt.show()