import cv2
import numpy as np

def grayscale(image):
    """converts image into grayscaled image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoising(image):
    """reduces noise in the image
    """
    return cv2.fastNlMeansDenoising(image)

def blur(image):
    """blurs image
    """
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)
    return dst
