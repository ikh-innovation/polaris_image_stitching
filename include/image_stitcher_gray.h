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

class ImageStitcherGray {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  ImageSubscriber left_sub_, right_sub_;
  image_transport::Publisher image_pub_;
  message_filters::Synchronizer<ApproxSyncPolicy> sync_;
  Mat homography_;
  bool show_image_;

public:
  ImageStitcherGray(const string& left_topic, const string& right_topic, int buffer_size, bool show_image = false)
    : it_(nh_),
      left_sub_(it_, left_topic, buffer_size, image_transport::TransportHints("compressed")),
      right_sub_(it_, right_topic, buffer_size, image_transport::TransportHints("compressed")),
      sync_(ApproxSyncPolicy(buffer_size), left_sub_, right_sub_),
      show_image_(show_image)
  {
    sync_.registerCallback(boost::bind(&ImageStitcherGray::callback, this, _1, _2));
    image_pub_ = it_.advertise("/stitched_images/output", 1);
    if (show_image_) namedWindow(OPENCV_WINDOW);
  }

  ~ImageStitcherGray() {
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

    // Convert to grayscale early
    Mat left_gray, right_gray;
    cvtColor(left_ptr->image, left_gray, COLOR_BGR2GRAY);
    cvtColor(right_ptr->image, right_gray, COLOR_BGR2GRAY);

    // Pad right image to allow warping result to fit
    copyMakeBorder(right_gray, right_gray, 0, 0, right_gray.rows, 0, BORDER_CONSTANT, Scalar(0));

    // Calibrate if needed
    if (homography_.empty()) calibrate(left_gray, right_gray);
    if (homography_.empty()) return;

    // Warp left grayscale image
    Mat warped_left;
    warpPerspective(left_gray, warped_left, homography_, right_gray.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    // Prepare stitched canvas (copy right image)
    Mat stitched = right_gray.clone();

    // Alpha blending
    int blend_width = 100;
    int blend_start = stitched.cols - blend_width - 200;
    blend_start = std::max(0, blend_start);
    Rect blend_roi(blend_start, 0, blend_width, stitched.rows);

    Mat roi_left = warped_left(blend_roi);
    Mat roi_right = stitched(blend_roi);

    Mat alpha_mask(roi_left.size(), CV_32FC1);
    for (int x = 0; x < alpha_mask.cols; ++x) {
      float alpha = static_cast<float>(x) / alpha_mask.cols;
      for (int y = 0; y < alpha_mask.rows; ++y)
        alpha_mask.at<float>(y, x) = alpha;
    }

    Mat roi_left_f, roi_right_f, blended_f;
    roi_left.convertTo(roi_left_f, CV_32FC1);
    roi_right.convertTo(roi_right_f, CV_32FC1);
    blended_f = roi_left_f.mul(1.0 - alpha_mask) + roi_right_f.mul(alpha_mask);

    Mat blended;
    blended_f.convertTo(blended, CV_8UC1);
    blended.copyTo(stitched(blend_roi));

    // Fill non-overlapping parts from warped_left (skip black)
    for (int y = 0; y < stitched.rows; ++y) {
      for (int x = 0; x < blend_start; ++x) {
        uchar val = warped_left.at<uchar>(y, x);
        if (val > 10) {
          stitched.at<uchar>(y, x) = val;
        }
      }
    }

    // Optional crop
    int crop_offset = 250;
    int final_width = stitched.cols - crop_offset;
    if (final_width <= 0) return;
    Mat cropped = stitched(Rect(crop_offset, 0, final_width, stitched.rows));

    // Publish
    cv_bridge::CvImage out_msg;
    out_msg.header = left_ptr->header;
    out_msg.encoding = sensor_msgs::image_encodings::MONO8;
    out_msg.image = cropped;
    image_pub_.publish(out_msg.toImageMsg());

    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }
};