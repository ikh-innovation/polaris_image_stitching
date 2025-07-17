# ROS Package for Image Stitching
Realtime Image Stitching for images obtained from two different unsynnchronized cameras.

# Usage

### Running the launch file:
```
roslaunch stitch_images stitch_images.launch 
```

### Running the launch file with Grayscale output:
```
roslaunch stitch_images stitch_images_gray.launch 
```

### Running rqt to view stitched image on topic /stitched_images/output
```
rosrun rqt_image_view rqt_image_view
```

#### Running the node using rosrun:
```
rosrun stitch_images stitch_images _left:=/camera_left/image_raw _right:=/camera_right/image_raw _image_transport_:=compressed
```

# Methodology for Image Stitching
1. Find BFM for both the streams
2. Find matching keypoints using BFMatcher
3. Compute Homography Trasformation
4. Use the computed homography to stitch the images

_The steps 1-3 are only performed for the first frames. Once the homography matrix has been calculated, we only use the calculated homography to stich the images together. For best result, rectified undistorted images should be used._