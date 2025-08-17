import cv2
image=cv2.imread("new_image.png")
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred_image=cv2.GaussianBlur(image,(5,5),0)
convervative_image=cv2.conservative