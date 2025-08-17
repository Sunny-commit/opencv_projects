import cv2 
import numpy as np

# Read the image
image = cv2.imread('new_image.png')

# Convert to grayscale
gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_scale, (5,5), 0)

# Perform High-Pass Filtering (extract edges)
high_pass = cv2.subtract(gray_scale, blurred_image)

# Convert blurred grayscale to 3-channel
blurred_colored = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

# Sharpening: Add high-pass details back to the original image
sharpened_image = cv2.addWeighted(image, 1.5, blurred_colored, -0.5, 0)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("High-Pass Filtered Image", high_pass)
cv2.imshow("Sharpened Image", sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
