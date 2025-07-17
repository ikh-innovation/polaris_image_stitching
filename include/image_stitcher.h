#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <opencv2/opencv.hpp>

static const std::string OPENCV_WINDOW = "Stitched Images";
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproxSyncPolicy;
typedef image_transport::SubscriberFilter ImageSubscriber;

using namespace cv;
using namespace std;

class ImageStitcher {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  ImageSubscriber left_sub_, right_sub_;
  image_transport::Publisher image_pub_;
  message_filters::Synchronizer<ApproxSyncPolicy> sync_;
  Mat homography_;
  bool show_image_;

public:
  ImageStitcher(const string& left_topic, const string& right_topic, int buffer_size, bool show_image = false)
    : it_(nh_),
      left_sub_(it_, left_topic, buffer_size, image_transport::TransportHints("compressed")),
      right_sub_(it_, right_topic, buffer_size, image_transport::TransportHints("compressed")),
      sync_(ApproxSyncPolicy(buffer_size), left_sub_, right_sub_),
      show_image_(show_image)
  {
    sync_.registerCallback(boost::bind(&ImageStitcher::callback, this, _1, _2));
    image_pub_ = it_.advertise("/robot/front_stitched/image_raw", 1);
    if (show_image_) namedWindow(OPENCV_WINDOW);
  }

  ~ImageStitcher() {
    if (show_image_) destroyWindow(OPENCV_WINDOW);
  }

  void calibrate(const Mat& img_left, const Mat& img_right) {
    Ptr<AKAZE> detector = AKAZE::create();
    vector<KeyPoint> kpts_left, kpts_right;
    Mat desc_left, desc_right;

    detector->detectAndCompute(img_left, noArray(), kpts_left, desc_left);
    detector->detectAndCompute(img_right, noArray(), kpts_right, desc_right);

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(desc_left, desc_right, knn_matches, 2);

    vector<DMatch> good_matches;
    for (const auto& match : knn_matches) {
      if (match[0].distance < 0.8f * match[1].distance)
        good_matches.push_back(match[0]);
    }

    if (good_matches.size() < 25) {
      ROS_WARN("Insufficient good matches for homography.");
      return;
    }

    vector<Point2f> pts_left, pts_right;
    for (const auto& m : good_matches) {
      pts_left.push_back(kpts_left[m.queryIdx].pt);
      pts_right.push_back(kpts_right[m.trainIdx].pt);
    }

    homography_ = findHomography(pts_left, pts_right, RANSAC);
  }

  void callback(const sensor_msgs::ImageConstPtr& left_msg, const sensor_msgs::ImageConstPtr& right_msg) {
    try {
      auto left_ptr = cv_bridge::toCvCopy(left_msg, sensor_msgs::image_encodings::BGR8);
      auto right_ptr = cv_bridge::toCvCopy(right_msg, sensor_msgs::image_encodings::BGR8);

      Mat right_img = right_ptr->image;
      copyMakeBorder(right_img, right_img, 0, 0, right_img.rows, 0, BORDER_CONSTANT, Scalar::all(0));

      if (homography_.empty()) calibrate(left_ptr->image, right_img);
      if (homography_.empty()) return;

      Mat warped_left;
      warpPerspective(left_ptr->image, warped_left, homography_, right_img.size(), INTER_LINEAR, BORDER_TRANSPARENT);

      Mat stitched = right_img.clone();

      int blend_width = 100;
      int blend_start = stitched.cols - blend_width - 200;
      blend_start = std::max(0, blend_start);

      Rect blend_roi(blend_start, 0, blend_width, stitched.rows);
      Mat roi_left = warped_left(blend_roi);
      Mat roi_right = stitched(blend_roi);

      // Create blend alpha mask
      Mat alpha_mask(roi_left.size(), CV_32FC1);
      for (int x = 0; x < alpha_mask.cols; ++x) {
        float alpha = static_cast<float>(x) / alpha_mask.cols;
        for (int y = 0; y < alpha_mask.rows; ++y)
          alpha_mask.at<float>(y, x) = alpha;
      }

      // Blend ROIs
      Mat left_f, right_f, alpha_3c;
      roi_left.convertTo(left_f, CV_32FC3);
      roi_right.convertTo(right_f, CV_32FC3);
      merge(vector<Mat>{alpha_mask, alpha_mask, alpha_mask}, alpha_3c);

      Mat blended_f = left_f.mul(1.0 - alpha_3c) + right_f.mul(alpha_3c);
      Mat blended;
      cv::threshold(blended_f, blended_f, 125.0, 125.0, THRESH_TRUNC);
      cv::threshold(blended_f, blended_f, 10.0, 10.0, THRESH_TOZERO);
      blended_f.convertTo(blended, CV_8UC3);
      blended.copyTo(stitched(blend_roi));

      // Fill non-overlapping warped_left onto stitched image
      Mat warped_channels[3];
      split(warped_left, warped_channels);

    // Fill non-overlapping part of left image (skip black/dark pixels)
    for (int y = 0; y < stitched.rows; ++y) {
      for (int x = 0; x < blend_start; ++x) {
        Vec3b val = warped_left.at<Vec3b>(y, x);
        if (val[0] > 10 || val[1] > 10 || val[2] > 10) {
          stitched.at<Vec3b>(y, x) = val;
        }
      }
    }

      // Crop left camera left side black 
      int crop_offset = 250;
      int final_width = stitched.cols - crop_offset;
      if (final_width <= 0) {
        ROS_WARN("Crop width invalid.");
        return;
      }

      Mat cropped = stitched(Rect(crop_offset, 0, final_width, stitched.rows));

      if (show_image_) {
        imshow(OPENCV_WINDOW, cropped);
        waitKey(3);
      }

      cv_bridge::CvImage out_msg;
      out_msg.header = left_ptr->header;
      out_msg.encoding = sensor_msgs::image_encodings::BGR8;
      out_msg.image = cropped;
      image_pub_.publish(out_msg.toImageMsg());

    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }
};