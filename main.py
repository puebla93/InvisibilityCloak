import cv2
from cvinput import cvwindows, Obj
import numpy as np
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Harry Potter's coat")
    parser.add_argument('-c', "--camera", dest="camera",type=int, default=0, help='Index of the camera to use. Default 0, usually this is the camera on the laptop display')
    parser.add_argument('-d', "--debug", dest="debugging",type=bool, default=False, help='Print all windows. This option is gor debugging')

    return parser.parse_args()

args = parse_args()

def main():
    # args = parse_args()

    camera = cvwindows.create('camera')
    capture = cv2.VideoCapture(args.camera)
    # capture.set(cv2.cv)

    #creating hsv window scale for range
    hsv_window = create_hsv_window()
    hue_sat = create_hue_sat_scale()
    new_hue_sat = draw_rectangle(hue_sat, hsv_window)
    hsv_window.show(new_hue_sat)
    
    print('sleeping 2 seconds')
    time.sleep(2)

    _, background = capture.read()

    # Flip the image
    background = cv2.flip(background, 1)

    while cvwindows.event_loop():
        
        _, image = capture.read()

        # Flip the image
        original_image = cv2.flip(image, 1)

        mask = get_mask(original_image, hsv_window)

        image_result = apply_mask(original_image, background, mask)

        camera.show(image_result)
        debug("original_image", original_image)

        new_hue_sat = draw_rectangle(hue_sat, hsv_window)
        hsv_window.show(new_hue_sat)

    cv2.destroyAllWindows()

def create_hsv_window():
    window = cvwindows.create("HSV")
    window.add_trackbar("HUE-Lower",default = 100, maxval = 180, allow_zero = True)
    window.add_trackbar("HUE-Upper",default = 133, maxval = 180, allow_zero = True)

    window.add_trackbar("SAT-Lower",default = 120, maxval = 255, allow_zero = True)
    window.add_trackbar("SAT-Upper",default = 255, maxval = 255, allow_zero = True)

    window.add_trackbar("VAL-Lower",default = 120, maxval = 255, allow_zero = True)
    window.add_trackbar("VAL-Upper",default = 255, maxval = 255, allow_zero = True)

    return window

def create_hue_sat_scale():
    hue_sat = np.zeros((256,181,3), np.uint8)

    sat = -1
    for x in range(256):
        hue = 0
        sat += 1
        for y in range(181):
            hue_sat[x,y,0] = hue
            hue_sat[x,y,1] = sat
            hue_sat[x,y,2] = 127
            hue += 1

    hue_sat = cv2.cvtColor(hue_sat, cv2.COLOR_HSV2BGR)
    return hue_sat

def draw_rectangle(image, window):
    x1 = window["HUE-Lower"]
    y1 = window["SAT-Lower"]
    x2 = window["HUE-Upper"]
    y2 = window["SAT-Upper"]
    new_image = image.copy()
    cv2.rectangle(new_image, (x1,y1), (x2, y2), (255, 255, 255), 1)

    x3 = (x1 + x2)/2
    y3 = (y1 + y2)/2
    h,s,v = image[y3,x3]

    val = np.zeros((256,100,3), np.uint8)
    
    for x in range(256):
        for y in range(100):
            val[x,y,0] = h
            val[x,y,1] = s
            val[x,y,2] = x
    new_val = val.copy()
    cv2.line(new_val,(0,window["VAL-Lower"]),(100,window["VAL-Lower"]),(255,255,255),1)
    cv2.line(new_val,(0,window["VAL-Upper"]),(100,window["VAL-Upper"]),(255,255,255),1)
    cv2.imshow('value', new_val)

    return new_image


def get_mask(original_image, hsv_window):
        image = cv2.GaussianBlur(original_image,(11,11), 0)
        debug('GaussianBlur', image)

        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([hsv_window["HUE-Lower"],hsv_window["SAT-Lower"],hsv_window["VAL-Lower"]])
        upper_blue = np.array([hsv_window["HUE-Upper"],hsv_window["SAT-Upper"],hsv_window["VAL-Upper"]])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        debug('mask', mask)

        kernel = np.ones((5,5),np.uint8)
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        erosion = cv2.erode(mask,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        mask = dilation

        debug('mask_erosion', mask)

        # gaussian_blur = cv2.GaussianBlur(mask,(5,5), 0)
        # _,thresh = cv2.threshold(gaussian_blur,100,255,cv2.THRESH_BINARY)
        # mask = thresh
        # cv2.imshow('GaussianBlur_mask', mask)

        median_blur = cv2.medianBlur(mask,5)
        _,thresh = cv2.threshold(median_blur,100,255,cv2.THRESH_BINARY)
        mask = thresh
        cv2.imshow('MedianBlur_mask', mask)

        return mask

def apply_mask(original_image, background, mask):
    # Bitwise-AND mask and background image
    bitwise_back_mask = cv2.bitwise_and(background, background, mask= mask)
        
    # Invert the mask image
    _, mask_inv = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)

    # Bitwise-AND mask and original image
    bitwise_image_mask_inv = cv2.bitwise_and(original_image, original_image, mask= mask_inv)

    # Sum bitwise_image_mask_inv and bitwise_back_mask images
    image_result = bitwise_back_mask + bitwise_image_mask_inv

    return image_result

def debug(window_name, image):
    if(args.debugging):
        cv2.imshow(window_name, image)


if __name__ == '__main__':
    main()