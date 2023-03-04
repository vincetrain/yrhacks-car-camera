# yrhacks-car-camera

<b>Make sure you download and place the yolov4.weights or yolov4-tiny.cfg file into this program's root directory!</b>  
[Download yolov4.weights here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)  
[Download yolov4-tiny.weights here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)  



<b>Quick Explaination</b>

This code is an implementation of a vehicle detection and speed measurement system using computer vision. Here is a brief summary of what the code does:

- It imports necessary libraries like time, numpy, cv2, pytesseract, and math.
- It sets up the confidence and NMS thresholds for object detection.
- It initializes the model and network for object detection using YOLOv4-tiny.
- It creates a window to display camera footage and resizes the window to fit the screen.
- It initializes two dictionaries for car tracking and car speeds.
- It starts an infinite loop that captures video from a specified source (in this case, a video file).
- Within the loop, it reads each frame from the video, detects objects using the YOLOv4-tiny model, and displays the detected objects along with their class names and     confidence scores on the video frame.
- It also uses the license plate recognition library (licenseocr) to get the license plate number of each detected vehicle and display it along with the vehicle class name and confidence score.
- It tracks each car by assigning a unique ID to each detected car and updating its position in each frame. It also calculates the speed of each car using its previous and current position and the time between frames.
- Finally, it displays the speed of each tracked car on the video frame and updates the window with each frame until the user presses 'q' to quit the program.
