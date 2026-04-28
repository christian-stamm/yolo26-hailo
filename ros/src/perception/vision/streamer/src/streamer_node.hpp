#ifndef STREAM_DEMO_NODE_HPP_
#define STREAM_DEMO_NODE_HPP_

#include <memory>
#include <string>
#include <deque>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vp_interface/msg/b_box_det_list.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

namespace livestream {

class StreamDemoNode : public rclcpp::Node {
 public:
  explicit StreamDemoNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~StreamDemoNode();

 private:
  // ROS 2 Publisher/Subscriber with zero-copy optimization
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bbox_image_pub_;

  // Message synchronizer with shared_ptr for zero-copy
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
  std::shared_ptr<message_filters::Subscriber<vp_interface::msg::BBoxDetList>> bbox_sub_;
  std::shared_ptr<message_filters::TimeSynchronizer<
    sensor_msgs::msg::Image,
    vp_interface::msg::BBoxDetList>> time_sync_;

  // Callbacks
  void frame_callback();
  void plot_frame_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
    const vp_interface::msg::BBoxDetList::ConstSharedPtr& bbox_msg);

  // Timer for frame publishing
  rclcpp::TimerBase::SharedPtr timer_;

  // Video capture
  cv::VideoCapture video_capture_;

  // Parameters
  double framerate_;
  std::string device_;
  std::string pub_video_topic_;
  std::string pub_detbox_topic_;
  std::string sub_detbox_topic_;
  int target_min_edge_px_;
  bool upscale_to_min_edge_;
  int frame_cache_size_;

  // Frame tracking
  int frame_num_;
  int num_frames_;

  // QoS for BEST_EFFORT
  rclcpp::QoS qos_profile_;

  struct CachedFrame {
    rclcpp::Time stamp;
    cv::Mat full_frame;
    int published_width;
    int published_height;
  };

  std::deque<CachedFrame> frame_cache_;
  std::mutex frame_cache_mutex_;

  // Helper functions
  void draw_bboxes(
    cv::Mat& frame,
    const vp_interface::msg::BBoxDetList& bbox_msg,
    double scale_x,
    double scale_y) const;

  cv::Mat resize_for_detector(const cv::Mat& frame) const;
  void cache_full_frame(
    const rclcpp::Time& stamp,
    const cv::Mat& full_frame,
    int published_width,
    int published_height);
  bool find_cached_frame(
    const rclcpp::Time& stamp,
    CachedFrame& out_frame);
};

}  // namespace livestream

#endif  // STREAM_DEMO_NODE_HPP_
