import cv2
import time
import numpy as np

# define the video capture object
cap = cv2.VideoCapture('yrhacks-car-camera/car-passing-by.mp4')

# define the parameters for object detection and tracking
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# initialize variables
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
prev_car_count = 0
car_count = 0
speed = 0
start_time = time.time()

# define the position of the line where we will track the cars
line_position = int(2 * height / 3)

# loop through the video frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # apply object detection and tracking
    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # get the bounding box of the object
        x, y, w, h = cv2.boundingRect(contour)
        
        # check if the object is a car by its size and position
        if w > 80 and h > 80 and y > line_position:
            # draw the bounding box around the car
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            car_count += 1
            
    # calculate the speed of the cars
    if car_count > prev_car_count:
        end_time = time.time()
        time_diff = end_time - start_time
        if time_diff > 0:
            speed = int((car_count / time_diff) * (60 * 60) / 1000)  # km/h
        start_time = end_time
        prev_car_count = car_count
    
    # display the frame
    cv2.putText(frame, "Speed: {} km/h".format(speed), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Traffic Cam', frame)
    
    # exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()