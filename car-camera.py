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

import cv2
import numpy as np
import skimage
import tensorflow
import imutils

cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    
    cv2.imshow('frame', frame)
    
    if cv2.wateKey(1) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()