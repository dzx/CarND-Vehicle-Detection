# -*- coding: utf-8 -*-
"""
Driver script to train a classifier and process video for vehicle tracking.
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from feature_lib import extract_features
from sklearn.model_selection import train_test_split
import pickle

from moviepy.editor import VideoFileClip
import os

cars = glob.glob("./classifier/vehicles/GTI_Far/*.png")
cars.extend(glob.glob("./classifier/vehicles/GTI_Left/*.png"))
cars.extend(glob.glob("./classifier/vehicles/GTI_MiddleClose/*.png"))
cars.extend(glob.glob("./classifier/vehicles/GTI_Right/*.png"))
cars.extend(glob.glob("./classifier/vehicles/KITTI_extracted/*.png"))

notcars = glob.glob("./classifier/non-vehicles/Extras/*.png")
notcars.extend(glob.glob("./classifier/non-vehicles/GTI/*.png"))

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_sample = mpimg.imread(cars[0])
non_car_sample = mpimg.imread(notcars[0])
fig, (left, right) = plt.subplots(ncols=2, figsize=(12, 9))
left.imshow(car_sample)
right.imshow(non_car_sample)
fig.tight_layout()
plt.show()

from feature_lib import get_hog_features
car_sample = cv2.cvtColor(car_sample, cv2.COLOR_RGB2YCrCb)
non_car_sample = cv2.cvtColor(non_car_sample, cv2.COLOR_RGB2YCrCb)
fig = plt.figure(figsize=(12,9))
car_feat, car_hog = get_hog_features(car_sample[:,:,0], orient, pix_per_cell, 
                                     cell_per_block,  vis=True, feature_vec=False)
plt.subplot("141")
plt.imshow(car_sample[:,:,0], cmap='gray')
plt.title("Car channel 1")
plt.subplot("142")
plt.imshow(car_hog, cmap='gray')
plt.title("Car HOG channel 1")
non_car_feat, non_car_hog = get_hog_features(non_car_sample[:,:,0], orient, pix_per_cell, 
                                             cell_per_block, vis=True, feature_vec=False)
plt.subplot("143")
plt.imshow(non_car_sample[:,:,0], cmap='gray')
plt.title("Non-car channel 1")
plt.subplot("144")
plt.imshow(non_car_hog, cmap='gray')
plt.title("Non-car HOG channel 1")
fig.tight_layout()


svc, X_scaler = None, None
try:
    with open("classifier.p", "rb") as clfile:
        classifier = pickle.load(clfile)
        if classifier != None:
            svc = classifier["cls"]
            X_scaler = classifier["scaler"]
except FileNotFoundError:
    from classifier_lib import train_car_classifier
    svc, X_scaler = train_car_classifier(cars, notcars, color_space, orient, pix_per_cell,
                                         cell_per_block, hog_channel, spatial_size,
                                         hist_bins, spatial_feat, hist_feat, hog_feat)
    classifier = {"cls":svc, "scaler":X_scaler}
    with open("classifier.p", "wb") as classifier_out:
        pickle.dump(classifier, classifier_out)
        classifier_out.close()

image = mpimg.imread("./test_images/test1.jpg")
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255

from feature_lib import get_hog_features, bin_spatial, color_hist
from search_lib import slide_window, search_windows, draw_boxes, find_cars

y_start_stop = [400, 656] # Min and max in y to search in slide_window()

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

windows.extend(slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)))
window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)

ystart = 400
ystop = 656
scale = 1.5
    
hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)

from search_lib import heat_filter

past_states = []
heat_state =  None

def process_image(image, diags=False):
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255
    global svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size 
    global hist_bins, heat_state, past_states
    ystart = 400
    ystop = 665
    hot_windows = []
    for scale in (1, 1.5):
        hot_windows.extend(find_cars(image, ystart, ystop, scale, svc, X_scaler, 
                                     orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    threshold = 1
    hist_len = 3
    if heat_state == None:
        heat_state = np.zeros_like(image[:,:,0]).astype(np.float)
    cons_windows, heat_state, heatmap = heat_filter(image, hot_windows, threshold, 
                                                    heat_state)
    past_states.append(heatmap)
    if len(past_states) > hist_len:
        heatmap = past_states.pop(0)
        heat_state = np.clip(heat_state - heatmap, 0, 255)
#    else:
#        cons_windows = heat_filter(image, hot_windows, threshold)
    window_img = draw_boxes(draw_image, cons_windows, color=(0, 0, 255), thick=6)                    
    if diags:
        return window_img, heat_state
    else:
        return window_img


for tst_fname in glob.glob("test_images/*.jpg"):
    tst_image = mpimg.imread(tst_fname)
    fig, (processed, heatm) = plt.subplots(ncols=2, figsize=(12, 9))
    fig.tight_layout()
    past_states = []
    heat_state =  None
    boxes, heat = process_image(tst_image, True)
    processed.imshow(boxes)
    heatm.imshow(heat) 
plt.show()

if not os.path.exists("test_videos_output"):
    os.mkdir('test_videos_output')
test_output = 'test_videos_output/test.mp4'
#clip1 = VideoFileClip("project_video.mp4")
clip1 = VideoFileClip("test_video.mp4")
past_states = []
heat_state =  None

for i in range(0, 20, 2):
    frame1 = process_image(clip1.get_frame(i))
    plt.imshow(frame1)
    plt.show()
 
past_states = []
heat_state =  None
   
test_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
test_clip.write_videofile(test_output,  audio=False)