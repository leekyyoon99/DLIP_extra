import cv2 as cv
import timeit as time
import torch
import datetime
import os
import socket
import pickle
from queue import Queue
import serial
import time

area = [100,150,225,300,375,450,525,600]
AREA_NUMBER = len(area)

def find_closest_index(target, arr = area):
    closest_diff = float('inf')
    closest_index = None
    for i, num in enumerate(arr):
        diff = abs(target - num)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i
    return closest_index

def calculate_weighted_index(arr):
    # all_sum = sum(arr)
    left_sum = sum(arr[:5])
    right_sum = sum(arr[7:])
    direction=0
    speed=-1
    if (left_sum>right_sum):
        direction=-1
        # speed = abs(4-index)
    elif(right_sum>left_sum):
        direction=1
        # speed = abs(5-index)
    return direction, speed

def send_array_to_arduino(array):
    array_string = ','.join(map(str, array))
    try:
        arduino.write((array_string + '\n').encode())
    except Exception as e:
        print(f"Error: {e}")

#load pretrained model
cap = cv.VideoCapture(2)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


#arduino
port = 'COM3'  
baudrate = 9600
arduino = serial.Serial(port, baudrate) 
time.sleep(3) #wait for arduino

# save img

if cap.isOpened():
    array_received=[]
    namevalue = []
    while True:
        state_list = [0 for _ in range(AREA_NUMBER)]
        #exit
        if cv.waitKey(1) & 0xFF == 27:
            break
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        #about model
        results= model(frame)
        person=[]
        for result in results.pandas().xyxy[0].iterrows():
            if (result[1]['name'] in ('person')):
                x1, y1, x2, y2 = int(result[1]['xmin']), int(result[1]['ymin']), int(result[1]['xmax']), int(result[1]['ymax'])
                person.append([x1,y1,x2,y2])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for plot in person:
            state_list[find_closest_index((plot[0]+plot[2])/2)]+=1
        #default
        array=[0,1]
        array[0], array[1] = calculate_weighted_index(state_list)
        print(array)
        
        send_array_to_arduino(array)

        cv.imshow("Webcam", frame)
else:
    print('cannot open the camera')
cap.release()
cv.destroyAllWindows()

