import cv2
import numpy as np

# Read the image
image = cv2.imread("new_image.png", cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

# Get dimensions
rows, columns = image.shape

# Number of pixels to add noise
no_of_pixels = np.random.randint(300, 10000)

# Add salt noise (white pixels)
for _ in range(no_of_pixels):
    y_coordinates = np.random.randint(0, rows)
    x_coordinates = np.random.randint(0, columns)
    image[y_coordinates, x_coordinates] = 255  # Set pixel to white

# Display images
cv2.imshow("Noise Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
