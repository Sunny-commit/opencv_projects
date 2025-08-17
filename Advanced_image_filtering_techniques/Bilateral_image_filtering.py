import cv2 as cv
image=cv.imread("new_image.png",cv.IMREAD_GRAYSCALE)
image=cv.resize(image,(255,255))
image=cv.cvtColor(image,cv.COLOR_RGB2BGR)
blurred_image=cv.bilateralFilter(image,20,200,300)
cv.imshow("blurred image",blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()
