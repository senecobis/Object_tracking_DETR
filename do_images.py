# Simple function to generate video from frames by Roberto Pellerito
import cv2
import os
import math

image_folder = "/home/roberto/old_trackformer/data/MULTI_PEOPLE/"
video_name = "/home/roberto/old_trackformer/data/MULTI_PEOPLE/multi_person.mov"

vidcap = cv2.VideoCapture(video_name)
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(image_folder + "frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1
