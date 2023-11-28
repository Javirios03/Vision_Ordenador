import cv2
import numpy as np

# We have 16 images of our pattern (piano keyboard)
path = "Calib_Images/"
images = []
for i in range(16):
    image = cv2.imread(path + f"image{i}" + ".jpg")
    images.append(image)

findCorners()
