import cv2
image=cv2.imread("new_image.png")
#Apply bilateral Filtering
bilateral_filtering=cv2.bilateralFilter(image,9,75,75)
cv2.imshow("Original Image",image)
cv2.imshow("Bilateral Image",bilateral_filtering)
from PIL import Image
bilateral_image=Image.fromarray(cv2.cvtColor(bilateral_filtering,cv2.COLOR_BGR2RGB))
bilateral_image.save("Bilateral_image_blur.png")
cv2.waitKey(0)
cv2.destroyAllWindows()