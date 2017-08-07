**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_example.jpg "Undistorted"
[image2]: ./output_images/warped_example.jpg "Road Transformed"
[image3]: ./output_images/find_line_sample.jpg "Threshold"
[image4]: ./output_images/found_line_with_window.jpg "Find Line"
[image5]: ./output_images/colored_line_area.jpg "Draw area"
[image6]: ./output_images/warped_colored_line_area.jpg "Warp Draw area"
[image7]: ./output_images/result_example.jpg "Fit Visual"
[image8]: /output_images/test2.jpg
[video1]: ./project_video.mp4 "Video"
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The chessboard images are used in camera calibration. 'cv2.findChessboardCorners' function is used find image points, i.e. the intersection of different color areas, on the chessboard, then 'cv2.calibrateCamera' is used for computing camera matrix and distortion coefficients based on object points define evenly on height and width of the image. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.

The code for this step is contained in 'cal_undistort' function in main.py



### Pipeline (single images)
The following is the detailed description of pipepline, applied on an example image like below
![alt text][image8]
#### 1. Distortion correction

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]

#### 2. Perspective transform

The code for my perspective transform appears in lines 1 through 8 in the file `example.py`.  Here the source and destination points are define as follows, in order to transform only the region of interest in origin image to a top-viewed image. 
<!-- ```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```
 --> 
Since the size of input image are fixed, perspective transform are the same for all images. Source and destination points are used in this project:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 480      | 200, 0        | 
| 290, 690      | 1080, 0       |
| 1074, 600     | 1080, 720     |
| 740, 480      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]
#### 3. Thresholding using color space and gradient.

I used a combination of color and gradient thresholds to generate a binary image, since hue and saturation channel in HLS performs robustly in detecting the lane line, these two color spaces are used in this section. I applied 'cv2.Sobel' on hue channel, then applied thresholding on absolute sobel value of hue channel and raw saturation channel. Combine the result of these two thresholded images as the output.
![alt text][image3]


#### 4. Find the line using window search
For an image, window search are used for the warped thresholded image. For a given window, select position on x-axis with most non-zero pixels as a point on a line. Go through the whole image from bottom to top.
For a video, do the same thing on the image of 1st frame, for later ones, only find non-zero pixels around the previous result line. If there is no line found or the newly found line has unreasonable curvature, redo the window search. The image below is an example of found lines.

![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

It is included in function 'measure_curvature'. The length of dashed line is 3.7m, the space between 2 dashed lines is about 20 feet = 6.1m, there are about 2 lines and 1 spaces in the image, the actual length covered in warped image is about 15 meters, which occupied 720 pixels. The warped 2 lane lines are about 900 pixels away from each other, with actual width of about 3.7 meters. Based on there two parameters, the curvature can be computed by the formula detailed explained at [here][http://www.intmath.com/applications-differentiation/8-radius-curvature.php].

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After find lines, applied perspective transformation again from warped image back to original image. Firstly, 'cv2.fillpoly' help draw the area detected to be between two lane lines on warped image. 
![alt text][image5]
then transforms the image back to original perspective.
![alt text][image6]
In the end, combine the lane line area with original image, result of the example image is shown as below

![alt text][image7]

---

### Pipeline (video)

#### 1. Test on video is 's a [link to my video result](./outputproject_video.mp4)

---

### Discussion

#### 1. I also test code on challenge and hard challenge video, the result is not satisfactory. The main reason for the challenge video is noise around the line, there is sharp linear change of the color of road surface around the line, therefore, this are likely to be passed the thresholding using gradient of color, and confuse the window search method to bring about incorrect result. 
I am trying to filter the noise out by using thresholding on grayscale image, in order to only reserve white and yellow pixels and discard dark ones, since the noise part consists of all dark and gray pixels. But this may cause new problem in test on the easy video, sometimes even lane line pixels are discarded after adding thresholding on gray image. This may due to grayscale pixels values vary under different light conditions. 
As for the hard challenge, there are more works required. Firstly, the perspective transform can not be the same for all images, since there are really sharp turn in which the starting and end points of 2 lane line may not lie in relatively middle of image. In this case, I think a preprocessing function is needed to decide the perspective transform in the first place. 
On the other hand, there are some frames in the hard challenge video in which the brightness is so high that the line pixels and road pixels are not distinguishable by eyes. If it only happens in a small portion around center image, it is can be ignored since it is the same as space between lane line, however, if it happens in a large portion, this may lead to fail to detect the line. I need to learn more to figure out how to fix this.

