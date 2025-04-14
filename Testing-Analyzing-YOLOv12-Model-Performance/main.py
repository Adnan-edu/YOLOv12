#----------Steps-------------------#
#Read an Image using OpenCV
#Load the YOLOv12 Model and Perform Object Detection
#Add the Confidence Value 'conf'
#Add NMS IOU 'iou'
#Add the classes parameter 'classes'
#Add the maximum detection parameter 'max_det
#Add show image parameter 'show = True'
#Add 'save_txt = True', save detection results in a text file
#Add 'save_crop = True' parameter
#Object Detection on Image
#Object Detection on Video and FPS Calculation
#----------------------------------#
#Import All the Required Libraries
import cv2
import math
import time
from ultralytics import YOLO
#Read the Image/Video/Live WebcamFeed using OpenCV
image = cv2.imread("Resources/Images/image1.jpg")
#Load the YOLOv12 Model
model = YOLO("yolo12n.pt")
#Object Detection using YOLOv12
# results = model.predict(image, save=True, conf=0.15, iou=0.1, classes=[0, 1], max_det=1) # Check conf score above 0.15 to draw boundary boxes
# results = model.predict(image, save=True, conf=0.15, iou=0.1, show=True) # Show only 1ms
# results = model.predict(image, save=True, conf=0.15, iou=0.1, show=True)
# results = model.predict(image, save=True, conf=0.15, iou=0.1, save_txt=True) # If you want to save bounding box coordinates into a text file
results = model.predict(image, save=True, conf=0.15, iou=0.1, save_txt=True, save_crop=True) # Save each detection in a separate image file
# Display the image using OpenCV
# cv2.imshow("Image", image)
# cv2.waitKey(0)