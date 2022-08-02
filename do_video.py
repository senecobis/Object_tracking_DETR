# Simple function to generate video from frames by Roberto Pellerito

import cv2
import os
import math

image_folder = "/home/roberto/old_trackformer/data/EXCAV/test"
video_name = "/home/roberto/ExcavVideo.mp4"

images = []
imges = sorted(os.listdir(image_folder))
for img in imges:
    images.append(img)
    print(img)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(video_name, fourcc, fps=25, frameSize=(width, height))


for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
