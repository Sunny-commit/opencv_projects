from PIL import Image
import cv2
image="new_image.png"
def convert_to_rgb(image_path):
    image=Image.open(image_path)
    if image.model!="RGB":
        image=image.convert("RGB")
    image.save("converte_rgb_image")
    image.show()
if __name__=="__main__":
    image_path="new_image.png"
    convert_to_rgb(image_path)