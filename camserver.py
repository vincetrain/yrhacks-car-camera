'''
TODO:
X Read data from camera / video 
X Computer vision to detect cars 
    - Calculate speed of each car and compare to local road's speed limit
    - Read license plate of each car and log the license,time,location
        - Only log if same data doesnt already exist
            - Log time data to nearest half hour?
    - Compare license to a database of stolen licenses
        - Alert a user? / log to new database if stolen license matches.
'''

import time
import numpy as np
import math
import cv2
import licenseocr

CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.2

VEHICLES = ['car', 'truck', 'motorbike', 'bus']

## initializes model and network for cars

class_file = open('./data/coco.names')
class_names = class_file.read().splitlines()

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# worse accuracy, but less intensive. uncomment if cuda gpu is disabled in cv2
model_config = 'cfg/yolov4-tiny.cfg'
model_weights = 'yolov4-tiny.weights'

# better accuracy, but more intensive. uncomment if cuda gpu is enabled in cv2
# model_config = 'cfg/yolov4.cfg'
# model_weights = 'yolov4.weights'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

## main loop

def calculate_speed(old_pos, new_pos, fps):
    # Calculate the distance between the two positions
    distance = math.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
    
    # Calculate the speed by dividing the distance by the time between frames
    speed = distance * fps
    
    return speed

car_tracker = {}
car_speeds = {}

while(True):
    cam = cv2.VideoCapture('examples/example-footage-1.mp4')

    while(cam.isOpened()):
        ret, frame = cam.read()
        cam_width, cam_height, _ = frame.shape
        if not ret:
            break
        
        start = time.time()
        
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for (class_id, score, box) in zip(classes, scores, boxes):
            # print(class_names[class_id], score, box)
            color = COLORS[int(class_id) % len(COLORS)]
            class_name = class_names[class_id]
            if (class_name in VEHICLES):
                plate, conf = licenseocr.get_plate(frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
                if plate == '' or conf < 20:
                    plate = 'License Not Found'
                label = "%s, %s : %f" % (class_name, plate, score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
                # check if a car is already being tracked
                if class_id in car_tracker:
                    # calculate speed
                    new_x, new_y, new_w, new_h = box
                    old_x, old_y, old_w, old_h, _ = car_tracker[class_id]
                    speed = calculate_speed((old_x, old_y), (new_x, new_y), 2)
                    car_speeds[class_id] = speed
                    # update tracker
                    car_tracker[class_id] = (new_x, new_y, new_w, new_h, time.time())
                else:
                    # add car to tracker
                    car_tracker[class_id] = (box[0], box[1], box[2], box[3], time.time())
                    car_speeds[class_id] = 0
            
        # display speed of each tracked car
        for car_id, speed in car_speeds.items():
            if speed is not None:
                label = "Speed: {} km/h".format(round(speed, 2))
                x, y, _, _ , _ = car_tracker[car_id]
                cv2.putText(frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        end = time.time()
        
        
        
        fps = 1 / (end-start)
        
        fps_label = f'Model FPS: {fps:.2f}'
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.namedWindow('Camera Footage', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera Footage', frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            exit(0)
        
        if key == ord(' '):
            while (cv2.waitKey(-1) != ord(' ')):
                print('Video paused. Press [SPACE] to unpause.')
        
    cam.release()

    cv2.destroyAllWindows()

