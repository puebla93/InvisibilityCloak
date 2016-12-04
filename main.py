import cv2
from cvinput import cvwindows, Obj
import sys

def main():
    camera = cvwindows.create('camera')
    capture = cv2.VideoCapture(0)

    # sys.sleep(5)
    print('sleeping 5 seconds')

    _, background = capture.read()

    while cvwindows.event_loop():
            _, image = capture.read()
            image = cv2.flip(image, 0)
            #process image
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            camera.show(image)

if __name__ == '__main__':
    main()