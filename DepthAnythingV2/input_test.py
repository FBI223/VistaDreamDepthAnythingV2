import cv2
img = cv2.imread("input.jpg")
print(img.shape)

print("Pierwsze 3 wartości piksela (0,0):", img[0,0])

from PIL import Image
import numpy as np

img_pil = Image.open("input.jpg")
arr = np.array(img_pil)
print(arr.shape)
print("Kanały:", arr[0,0])
