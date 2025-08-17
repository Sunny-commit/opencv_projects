import cv2
import numpy as np
image=cv2.imread("new_image.png")
#Convert image to GrayScale
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#Apply Gaussian smoothing to create a blurred version of the image
blurred_image=cv2.GaussianBlur(gray_image,(5,5),0)
sharpened_image=cv2.addWeighted(gray_image,1.5,blurred_image,-0.5,0)
#display the original and sharpend image
from PIL import Image
colored_sharpened_image=cv2.cvtColor(sharpened_image,cv2.COLOR_BGR2RGB)
cv2.imshow("Original image",image)
cv2.imshow("unsharp masking",colored_sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()