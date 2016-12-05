import cv2
from cvinput import cvwindows, Obj
import numpy as np
import time

def main():
    camera = cvwindows.create('camera')
    capture = cv2.VideoCapture(0)

    print('sleeping 5 seconds')
    time.sleep(5)

    _, background = capture.read()

    # Flip the image
    background = cv2.flip(background, 1)

    while cvwindows.event_loop():
        
        _, image = capture.read()

        # Flip the image
        image = cv2.flip(image, 1)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and background image
        bitwise_back_mask = cv2.bitwise_and(background, background, mask= mask)
        
        # Invert the mask image
        _, mask_inv = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)

        # Bitwise-AND mask and original image
        bitwise_image_mask_inv = cv2.bitwise_and(image, image, mask= mask_inv)

        # Sum bitwise_image_mask_inv and bitwise_back_mask images
        image_result = bitwise_back_mask + bitwise_image_mask_inv

        camera.show(image_result)

if __name__ == '__main__':
    main()