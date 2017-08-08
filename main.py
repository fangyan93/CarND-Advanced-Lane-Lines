import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import sys
import math
SHOW = False
def cal_undistort(img):
	"""
	create objpoints and find corners in a chessboard image
	"""
	objpoints = []
	imgpoints = []
	nx = 9
	ny = 6
	objp = np.zeros((nx*ny, 3), np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	if ret == False:
		print("No corner found!")
		return [],[],[], res
	imgpoints.append(corners)
	objpoints.append(objp)
	ret1, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist, mtx, dist, ret

img = cv2.imread("./camera_cal/calibration2.jpg")
undistorted, mtx, dist, found = cal_undistort(img)

if found == False:
	exit(1)

test_img = cv2.imread("./test_images/test2.jpg")
offset = 200
shape = test_img.shape
img_size = shape[0:2]
img_size = img_size[::-1]

src = np.float32([[540,480], [740,480], [1075,690], [290,690]])
dst = np.float32([[offset, 0],[shape[1] - offset, 0], [shape[1] - offset, shape[0]], [offset, shape[0]]])

# set 
M = cv2.getPerspectiveTransform(src, dst)
M_reverse = cv2.getPerspectiveTransform(dst, src)

def thresholding(img, s_thresh = (120, 255), grad_thresh = (20,120)):
	"""
	Apply thresholding on color pixels and its gradients
	"""
	# combination of color and gradient threshing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray_thresh = (70, 255)
	# apply thresholding on grayscale image first in order to remove noise around the line.
	threshed_g = np.zeros_like(gray)
	threshed_g[(gray > gray_thresh[0]) & (gray <= gray_thresh[1])] = 1
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	sobelx = abs(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
	scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))

	threshed_l = np.zeros_like(scaled_sobel)
	threshed_l[(scaled_sobel > grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1

	threshed_s = np.zeros_like(scaled_sobel)
	threshed_s[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	threshed = threshed_s | threshed_l
	threshed &= threshed_g
	
	return threshed

def find_line(binary_warped):
	"""
	Find line in warped thresholded image using window search
	"""
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	midpoint = np.int(histogram.shape[0]/2)
	center_offset = 50
	leftx_base = np.argmax(histogram[:midpoint-50])
	rightx_base = np.argmax(histogram[midpoint+50:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    if SHOW == True:
	    	cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    	cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	plt.
	if SHOW == True:
		plt.plot(left_fitx, ploty, color='blue')
		plt.plot(right_fitx, ploty, color='blue')
		# plt.imshow(out_img)
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.savefig('found_line_with_window.jpg', bbox_inches='tight')
		plt.show()
		
	return out_img, left_fitx, right_fitx, leftx, rightx, lefty, righty, left_fit, right_fit, ploty

def skip_window_search(binary_warped, left_fit, right_fit):
	"""
	For video, search around found lines in previous frames in order to be more efficient, smooth and robust.
	"""
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# only search around the previous found line with a margin
	margin = 50
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	if len(leftx) == 0:
		plt.imshow(binary_warped)
		plt.show()
		return binary_warped, [], [], [], [], [], [],[],[],[]
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	return binary_warped, left_fitx, right_fitx, leftx, rightx, lefty, righty, left_fit, right_fit, ploty


def draw_line(out_img, left_fitx, right_fitx, ploty, undistorted):
	"""
	Apply result of find_line on original image
	"""
	warp_zero = np.zeros(out_img.shape[0:2]).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, M_reverse, img_size) 
	
	if( SHOW == True) :
		f, (ax1, ax2) = plt.subplots(1,2,figsize=(24,9))
		ax1.imshow(newwarp)
		ax1.set_title("test image 2")
		ax2.imshow(undistorted)
		ax2.set_title("test image 3")
		plt.show()
		cv2.imwrite('colored_line_area.jpg',color_warp)
		cv2.imwrite('warped_colored_line_area.jpg',newwarp)
	# Combine the result with the original image
	result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
	
	return result

def measure_curvature(left_fitx, right_fitx, ploty):
	"""
	Measure approximate curvature of found lines
	"""
	# the length of dashed line is 3.7m, the space between 2 dashed lines is about 20 feet = 6.1m, there are about 4 lines and spaces in the image
	ym_per_pix = 15/720 
	# the warped 2 lane lines are about 900 pixels away from each other
	xm_per_pix = 3.7/900 
	y_eval = max(ploty)

	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	print(left_curverad, 'm', right_curverad, 'm')
	return left_curverad, right_curverad


def find_line_in_image(img):
	''' Use functions above to find lane line in an image
		Input: the original image
		Output: the input image with found line added
	'''
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
	warped = cv2.warpPerspective(undistorted, M, img_size)
	threshed = thresholding(warped)
	# specify global variables
	global first_time
	global left_fit
	global right_fit
	global right_fitx
	global left_fitx
	global ploty
	global count
	# for first frame
	if first_time == False:
		first_time = True
		out_img, left_fitx, right_fitx, leftx, rightx, lefty, righty, left_fit, right_fit, ploty = find_line(threshed)
	else:
		out_img, left_fitx, right_fitx, leftx, rightx, lefty, righty, left_fit, right_fit, ploty = skip_window_search(threshed, left_fit, right_fit)
		# if no line found, use previous found line
		if(len(left_fitx) > 0):
			left_curve, right_curve = measure_curvature(left_fitx, right_fitx, ploty)
			# if curvature is unreasonable, increment count
			if left_curve == right_curve or min(left_curve, right_curve) < 100 or max(left_curve, right_curve) > 150000:
				count += 1
				print("Bad %d!" % count)
			else:
				count = 0
			# if continues problematic line finding encountered, redo window search
			if count > 2:
				out_img, left_fitx, right_fitx, leftx, rightx, lefty, righty, left_fit, right_fit, ploty = find_line(threshed)
				count = 0
	result = draw_line(out_img, left_fitx, right_fitx, ploty, undistorted)
	if SHOW == True:
		cv2.imwrite('find_line_sample.jpg',out_img)
		cv2.imwrite('result_example.jpg',result)
		cv2.imwrite('undistorted_example.jpg',undistorted)
		cv2.imwrite('warped_example.jpg',warped)
		cv2.imwrite('threshed_example.jpg',threshed)
		plt.imshow(result)
		plt.show()
	
	lcurve_string = 'Left Curvature: ' + str(left_curve) + 'm'
	rcurve_string = 'Right Curvature: ' + str(right_curve) + 'm'
	# x coordinate of mid point of bottom of region interest is (1075 + 290) / 2 = 682.5
	mid = (682.5 - (leftx[0] + rightx[0]) / 2) * (3.7 / 9)
	dis_to_center = 'Distance from center: ' + str(mid) + ' cm'
	cv2.putText(result, lcurve_string, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=2)
	cv2.putText(result, rcurve_string, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=2)
	cv2.putText(result, dis_to_center, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=2)
	return result

import os
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML

video_path = ""
def find_in_video(video_name):
	"""
	Process video using VideoFileClip, apply find_line_in_image for each frame in video and create video of results
	Input: name of directory
	Output: None
	"""
	input_video = video_name
	output_video = "output" + video_name
	white_output = os.path.join(video_path, output_video)
	white_input = os.path.join(video_path, input_video)
	clip1 = VideoFileClip(white_input)
	white_clip = clip1.fl_image(find_line_in_image) 
	white_clip.write_videofile(white_output, audio=False)
	HTML("""
    <video width="960" height="540" controls>
      <source src="{0}">
    </video>
    """.format(white_output))


def for_lines(directory):
	"""
	Run find_line_in_image for all images in a directory
	Input: name of directory
	Output: None
	"""
	files = os.listdir(directory)
	for file in files:
		if(file == '.DS_Store' or file == "."):
		    continue
		print(file)
		file = os.path.join(directory, file)
		image = cv2.imread(file)
		result = find_line_in_image(image)
		plt.imshow(result)
		plt.show()

# set the undistortion coefficient and camera matrix as global variable

# set several global variable which will be updated in current frame and used in future frames for test on video
first_time = False
left_fit = None 
right_fit = None 
left_fitx = None
right_fitx = None
ploty = None
count = 1

if __name__ == "__main__":
	img_example = cv2.imread("./test_images/test2.jpg")
	find_line_in_image(img_example)
	# find_in_video("project_video.mp4")




