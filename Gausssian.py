import cv2
import matplotlib.pyplot as plt
image=cv2.imread("new_image.png")
blurred=cv2.GaussianBlur(image,(5,5),0)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1),plt.imshow(image)
plt.title("original image")
plt.subplot(1,2,2),plt.imshow(blurred,cmap='gray')
plt.title("Gaussina image")
plt.show()