import cv2
import matplotlib.pyplot as plt
image=cv2.imread("new_image.png")
blurred_image=cv2.GaussianBlur(image,(5,5),0)
#cv2.imshow("Original Image",image)
#cv2.imshow("Gaussian Smoothing",blurred_image)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1),plt.imshow(image),plt.title("Original Image")
plt.subplot(1,2,2),plt.imshow(blurred_image),plt.title("Gaussian_Smoothing Image")

cv2.waitKey(0)
cv2.destroyAllWindows()