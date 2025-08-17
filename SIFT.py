#Scale-invariant Feature Transform:In this we can use to adjust the image 
import cv2
import matplotlib.pyplot as plt
sift=cv2.SIFT_create()

image=cv2.imread("new_image.png")
print(image.shape)
keypoints,descriptors=sift.detectAndCompute(image,None)
sift_image=cv2.drawKeypoints(image,keypoints,None)
plt.imshow(sift_image,cmap='gray')
plt.title("SIFT keypoints")
plt.show()