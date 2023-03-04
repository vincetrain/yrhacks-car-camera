'''
Todo:
- Read data from camera / video input
- Computer vision to detect cars
    - Calculate speed of each car and compare to local road's speed limit
    - Read license plate of each car and log the license,time,location
        - Only log if same data doesnt already exist
            - Log time data to nearest half hour?
    - Compare license to a database of stolen licenses
        - Alert a user? / log to new database if stolen license matches.
'''

DEV = True

import cv2
import numpy as np
import skimage
import tensorflow
import imutils

def get_plate(self, frame):
    '''
    Returns coordinates of license plate given a frame.
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    _, contours, _ = cv2.findContours(thresh.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)


## main loop

if (DEV):
    cam = cv2.VideoCapture('example-footage.mp4')
else:
    cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    
    display = imutils.resize(frame, width=1080)
    cv2.imshow('video footage', display)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    if key == ord(' '):
        while (cv2.waitKey(-1) != ord(' ')):
            print('Video paused. Press [SPACE] to unpause.')
    
cam.release()

cv2.destroyAllWindows()

