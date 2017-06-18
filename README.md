# **Vehicle Detection Project**

## The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/train_sample.png
[image2]: ./output_images/hog_sample.png
[image3]: ./output_images/windows.png
[image4]: ./output_images/pipe_out_1.png
[image5]: ./output_images/pipe_out_2.png
[image6]: ./output_images/pipe_out_3.png
[image7]: ./output_images/pipe_out_4.png
[image8]: ./output_images/pipe_out_5.png
[image9]: ./output_images/pipe_out_6.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `feature_lib.py` file, function `extract_features` which calls rest of the functions in said file.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
Knowing what kind of extracted features I get to work with (HOG, color histograms, spatial features) I used following take-away points from
project lectures as well as data science postulates as guidlines for my experimentation with smaller subsample of training set:

* Some color spaces work much better than others for given dataset. Find the one that performs best with other model parameters fixed, and build on that.
* HOG has highest predictive power followed by color histogram and finaly raw spatial features.
* Adding features may increas model accuracy, but it certainly increases training time and some features may actualy hinder the model. In light of that, it is important to know that...
* Predictive power of HOG tends to increase with number of directions up to 9 directions. Adding more directions doesnt seem to help.

Then I wrote feature extraction pipeline, in file `feature_lib.py` which includes `get_hog_features` function.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
Considering the above I arrived to `YCrCb` color space with 9 directions and all channels used for HOG. 
For remaining parameters and feature set I mostly started with lecture defaults and quickly arrived to test accuracy of 0.9901.
So I decided to leave in questionable features such as spatial as accuracy is probably high enough to move on to next stage of project.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using SKLearn library on top of training set provided by Udacity. Number of positive and negative samples was pretty balanced, so I tried to use the 
dataset without augmentation. As described above, I mostly focused on feature extraction parameters, and that alone resulted in accuracy beyond 99% without any experimentation
with different SVM kernels and regularization parameters. So in my experience this is already well within the zone of diminishing returns. SVM training code is in `classifier_lib.py`


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I identified region of interest being (400-655) across Y-axis and entire X-range. I used blocks sized 64x64 as that is size of training images
as well as 96x96 as that is the apparent size of cars in video when they are within couple of feet. 

I also tred sizes 32x32 and 24x24 in reduced ROI (400-500),
in attempt to get some matches on cars that are far ahead on the road. However I gave up on that, as I was only getting many false positives and not
a single match from that set of windows, while it slowed down the pipeline due to processing extra windows. I started with 50% window overlap and
stuck to it as it seems to work well enough.

Generation of search windows happens in `search_lib.py` around lines 142-147 where it is tighlty coupled with extracting HOG for entire ROI and
then applying generated windows to extracted features and eliminating negatives. Therefore I dont have full set of generated windows from the
actual `find_cars` function, but this should be a very close approximation:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Since classification needs to work on same set of features that was used for training, I used feature extraction as above,  namely two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color. Result needed some duplicate elimination, so I used heatmap thresholding and binning 
in order to clip spurious detections and identify blobs on heatmap using `scipy.ndimage.measurements.label()`. These blobs remaining after thesholding are assumed to be distinct cars and bounding rectangles are constructed around them. Below is the pipeline output for test images.

![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output/test.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Filter for false positives on a single frame is described above. However, video processing pipeline records individual heatmaps for last 3 frames, and adds them together applying higher threshold to it. The rest of heatmap blob labeling is then done the same way as for 
single frame processing. Old frames are subtracted from cumulative heatmap when they are evicted from buffer, so that cumulative heatmap allways
retains exact sum of heatmaps. This somewhat helps eliminate most false positives without decimating true positives in some frames.  

### Six frames out of pipeline with their corresponding integrated heatmaps are provided above




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Since lectures provide guidance on how to do appropriate feature engineering for this particular machine learning application, it didn't take much to arrive to classifier that
is over 99% accurate. But that still leaves room for false postitives and false negatives. While I greatly reduced the number of actual false positives by reducing ROI to the bottom of image
where the road is at, classifier just wasn't picking up any cars that are far ahead on the road and appear to fit in 16x16 or 24x24 rectangles. So clasifier could use more work in order to 
get more robust with images of different sizes. And there is lot of room for improvements since it is really just SVM with linear kernel and no parameter tuning was done yet.

Also, I used heatmap and blob labeling in order to filter out false positives and stabilize the size of bounding rectangles around detected cars. This approach worked to some point as maintaining
cumulative heatmap over too many frames required higher thresholds and this led to less responsive formation and movement of bounding rectangles. Additional improvement would require tracking actual
detected cars from frame to frame, but since I started this project one week behind the schedule, there was no time to implement this.


