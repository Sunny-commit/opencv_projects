import cv2 
import numpy as np

# Read the image
image = cv2.imread('new_image.png')

# Convert to grayscale
gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian_image=cv2.Laplacian(gray_scale,cv2.CV_64F)

#Normalize the output to make it suitable for display
laplacian_image=cv2.normalize(laplacian_image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
cv2.imshow("original",gray_scale)
cv2.imshow("Laplacian Sharpening",laplacian_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
