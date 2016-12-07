import cv2
from cvinput import cvwindows, Obj
import numpy as np
import time

def main():
    camera = cvwindows.create('camera')
    capture = cv2.VideoCapture(0)

    hsv_lower = cvwindows.create("HSV_LOWER")

    hsv_lower.add_trackbar("HUE-Lower",default = 90, maxval = 360, allow_zero = True)
    hsv_lower.add_trackbar("SAT-Lower",default = 120, maxval = 255, allow_zero = True)
    hsv_lower.add_trackbar("VAL-Lower",default = 40, maxval = 255, allow_zero = True)
    
    color_lower = np.zeros((50,400,3), np.uint8)
    color_lower[:] = [hsv_lower["HUE-Lower"],hsv_lower["SAT-Lower"],hsv_lower["VAL-Lower"]]
    color_lower = cv2.cvtColor(color_lower, cv2.COLOR_HSV2BGR)
    hsv_lower.show(color_lower)
    
    hsv_upper = cvwindows.create("HSV_UPPER")

    hsv_upper.add_trackbar("HUE-Upper",default = 130, maxval = 360, allow_zero = True)
    hsv_upper.add_trackbar("SAT-Upper",default = 255, maxval = 255, allow_zero = True)
    hsv_upper.add_trackbar("VAL-Upper",default = 255, maxval = 255, allow_zero = True)
    
    color_upper = np.zeros((50,400,3), np.uint8)
    color_upper[:] = [hsv_upper["HUE-Upper"],hsv_upper["SAT-Upper"],hsv_upper["VAL-Upper"]]
    color_upper = cv2.cvtColor(color_upper, cv2.COLOR_HSV2BGR)
    hsv_upper.show(color_upper)

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
        lower_blue = np.array([hsv_lower["HUE-Lower"],hsv_lower["SAT-Lower"],hsv_lower["VAL-Lower"]])
        upper_blue = np.array([hsv_upper["HUE-Upper"],hsv_upper["SAT-Upper"],hsv_upper["VAL-Upper"]])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cv2.imshow('mask', mask)

        kernel = np.ones((15,15),np.uint8)
        dilation = cv2.dilate(mask,kernel,iterations = 1)
        erosion = cv2.erode(dilation,kernel,iterations = 1)
        mask = erosion

        # Bitwise-AND mask and background image
        bitwise_back_mask = cv2.bitwise_and(background, background, mask= mask)
        
        # Invert the mask image
        _, mask_inv = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)

        # Bitwise-AND mask and original image
        bitwise_image_mask_inv = cv2.bitwise_and(image, image, mask= mask_inv)

        # Sum bitwise_image_mask_inv and bitwise_back_mask images
        image_result = bitwise_back_mask + bitwise_image_mask_inv


        color_lower[:] = [hsv_lower["HUE-Lower"],hsv_lower["SAT-Lower"],hsv_lower["VAL-Lower"]]
        color_lower = cv2.cvtColor(color_lower, cv2.COLOR_HSV2BGR)
        hsv_lower.show(color_lower)

        color_upper[:] = [hsv_upper["HUE-Upper"],hsv_upper["SAT-Upper"],hsv_upper["VAL-Upper"]]
        color_upper = cv2.cvtColor(color_upper, cv2.COLOR_HSV2BGR)
        hsv_upper.show(color_upper)

        camera.show(image_result)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()