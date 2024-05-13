import cv2
import numpy as np

hog = cv2.HOGDescriptor()

def extract_hog_128(img):
    img = img.resize((128, 128))
    return hog.compute(np.array(img))

