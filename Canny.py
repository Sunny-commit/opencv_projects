import cv2
import matplotlib.pyplot as plt
image=cv2.imread("new_image.png")
edges=cv2.Canny(image,50,150)
plt.figure(figsize=(200,100))
plt.subplot(1,2,1),plt.imshow(image)
plt.subplot(1,2,2),plt.imshow(edges)
plt.show()


