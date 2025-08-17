import cv2
from PIL import Image

# Read the image
image = cv2.imread("new_image.png")

# Apply median filtering
median_filtered_image = cv2.medianBlur(image, 5)

# Display the original and median filtered image
cv2.imshow("Original Image", image)
cv2.imshow("Median Filtering", median_filtered_image)

# Convert the NumPy array to a PIL Image and save
median_pil_image = Image.fromarray(cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB))
median_pil_image.save("Median_Filtered_image.png")  # Ensure file extension is included

cv2.waitKey(0)
cv2.destroyAllWindows()
