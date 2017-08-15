** Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Additionally, apply a color transform and append binned histograms of color to the HOG feature vector.
* Normalize the features and train a classifier using the random forest classifier. Determine the best set of hyper-parameters using cross validation.
* Implement a sliding-window and use the trained classifier to search for vehicles in images.
* Run the image processing pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/preview.png "Training data preview"
[image2]: ./images/HLS_vehicle.png "HSL space image"
[image3]: ./images/color_hist.png "Color space histogram"
[image4]: ./images/HOG_example.png "Example of HOG features"
[image5]: ./images/feature_scaling.png "Feature scaling"
[image6]: ./images/sliding_window.png "Example results"
[image7]: ./images/bound_box.png "Heatmap and bounding box example"
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Training images

I started by reading in all the `vehicle` and `non-vehicle` training images.  The labeled data set of vehicle and non-vehicle images are downloaded from [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). Each class has over 8000 images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Color features

I then explored different color channels to see if the vehicles can still be identified with a single color channels. For example, after converting the HLS channels, the S channels still contains enough information for human to identify vehicles.

![alt text][image2]

Therefore, the first few features are the color space histograms using the Red, Green, Blue channels in the RGB space and the L and S channels in the HLS space. The code to extract those features is located in function `get_color_hist()`. Here I am resizing all images to 32 x 32 before extracting the color histograms. The following are color histograms extracted for a vehicle image and a non-vehicle image.

![alt text][image3]

### Histogram of Oriented Gradients (HOG) features

#### 1. Extracting HOG features from the training images.

The code for this step is contained in the function `get_HOG_features()`. Depending on the input parameters, this function will return the extracted HOG features directly from a gray image, or the concatenated HOG features of all three color channels.

I then explored different parameters for (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the grayscale images and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

#### 2. Choice of HOG parameters.

I tried different combinations of parameters and settled with the parameter set `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. For a 64 x 64 image, this gives 7 x 7 x 2 x 2 x 8 = 1568 features in each color channel, and I used RGB channels here.

#### 3. Combining the color histogram features and HOG features.

The function `extract_features()` extracts the color histograms and HOG features, and combines them into one feature vector. Because different features have different value ranges, I used `StandardScaler` function in sklearn to scale all features to have 0 mean and unit deviation. The following graph shows the feature amplitudes before and after scaling.

![alt text][image5]

#### 4. Train a classifier using the selected HOG features and color features to distinguish vehicles and non-vehicles.

I trained a random forest classifier for this classification. Random forest allows trade-off between accuracy and running time by choosing the `n_estimators` and `max_features` values. Other parameters are determined using `RandomizedSearchCV()` in sklearn with a 3-fold cross validation. The final set of parameters is:

`{'max_features': 0.15, 'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 0.0001}`

This gives an average validation accuracy of 0.9805.

### Sliding Window Search

#### 1. Implementation of the sliding window search.

The sliding window search is implemented in function `slide_window()` and the returned window list is passed to function `predict_windows()` to extract the features from all windows, scale the features using the scaler, and identify windows with cars using the classifier. I used window sizes `[64, 96, 128, 192]` to accommodate different car sizes caused by different distances from the camera, and an overlap of 0.75 to ensure enough coverage for all possible locations. Since cars running on the road all appear at the lower half of the image, I am also only searching at the lower half.

#### 2. Optimization of the image pipeline.

Because the searching windows have large overlap, extracting the HOG features from each window separately involves lots of repeated calculations. Function `find_vehicle_windows()` uses a more efficient way to extract HOG features by:
1. Extract the HOG features over the entire area of interest.
2. Select the relevant HOG features from the overall HOG map during sliding window search.

With 0.75 overlap in sliding windows, this saves 75% calculation used for HOG feature extraction.

#### 3. Examples of test images to demonstrate the pipeline.

Ultimately I searched on four window sizes to extract all visible cars.  Here are some example results:

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Filter for false positives and combining overlapping bounding boxes.

Since the cars cannot suddenly appear or disappear in the stream, I recorded the positions of positive detections in the last 10 frames of the video. From the positive detections I created a heatmap and then thresholded that map to identify possible vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, each blob corresponded to a vehicle. Then I constructed bounding boxes to cover the area of each blob detected.  

Here is an example showing the heatmap from an individual image, the labeled blobs from the heatmap, and the bounding boxes overlaid on the image.

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem is with the features. Here I only used color features in RGB and HLS spaces, and HOG features in RGB space. This is not working very well when there is too much or too little illumination. In the output video, I lost track of the white car when passing the bright region and lost track of the black car when passing the shadow region. Exploring more color spaces may help solve this problem.

The second problem is with the classifier. It works well for vehicle detection, but tend to return positive result with the vehicle only covers a small part of the image. Therefore, when doing sliding window search, cars far away will yield positive values in a large area of heatmap, which leads to a large bounding box, but in fact a smaller bounding box will be more useful in real applications. Augmenting the training dataset may also solve this problem.

The third problem is *false* positives. In the video stream, there are many cars coming from the opposite direction that we do not need to detect. Those areas are determined as car areas in the output video. One way to remove those boxes is to limit the window search region to the right side, but this will make the pipeline fail when the car is not driving on the leftmost lane. Another possible way is to augment the training dataset so that cars behind the fences will be labeled as non-vehicles.
