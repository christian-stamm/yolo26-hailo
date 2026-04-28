#include "streamer_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace livestream {

StreamDemoNode::StreamDemoNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("StreamDemo", options),
      frame_num_(0),
      num_frames_(0),
      qos_profile_(rclcpp::SensorDataQoS()) {
  
  // Configure QoS for BEST_EFFORT with depth 1
  qos_profile_.best_effort();
  qos_profile_.keep_last(1);

  // Declare and get parameters
  framerate_ = this->declare_parameter("pub_video_rate", 30.0);
  device_ = this->declare_parameter<std::string>("pub_video_source", "0");

  std::string pub_video_topic_prefix = this->declare_parameter<std::string>(
      "pub_video_topic_prefix", "/camera");
  std::string pub_video_topic_suffix = this->declare_parameter<std::string>(
      "pub_video_topic_suffix", "/image_raw");
  
  std::string sub_detbox_topic_prefix = this->declare_parameter<std::string>(
      "sub_detbox_topic_prefix", "/yolo26");
  std::string sub_detbox_topic_suffix = this->declare_parameter<std::string>(
      "sub_detbox_topic_suffix", "/detstream");
  std::string pub_detbox_topic_suffix = this->declare_parameter<std::string>(
      "pub_detbox_topic_suffix", "/image_bbox");

  target_min_edge_px_ = this->declare_parameter("detector_min_edge_px", 640);
  upscale_to_min_edge_ = this->declare_parameter("upscale_to_min_edge", false);
  frame_cache_size_ = this->declare_parameter("frame_cache_size", 60);

  if (target_min_edge_px_ <= 0) {
    RCLCPP_WARN(this->get_logger(), "detector_min_edge_px must be > 0. Falling back to 640.");
    target_min_edge_px_ = 640;
  }
  if (frame_cache_size_ <= 0) {
    RCLCPP_WARN(this->get_logger(), "frame_cache_size must be > 0. Falling back to 60.");
    frame_cache_size_ = 60;
  }

  // Build topic names
  pub_video_topic_ = pub_video_topic_prefix + pub_video_topic_suffix;
  sub_detbox_topic_ = sub_detbox_topic_prefix + sub_detbox_topic_suffix;
  pub_detbox_topic_ = sub_detbox_topic_prefix + pub_detbox_topic_suffix;

  // Validate framerate
  if (framerate_ <= 0.0) {
    RCLCPP_ERROR(this->get_logger(), "Framerate must be greater than 0, got: %f", framerate_);
    throw std::invalid_argument("framerate must be greater than 0");
  }

  // Convert device to int if it's numeric, otherwise use as string (device path)
  try {
    int device_id = std::stoi(device_);
    video_capture_.open(device_id);
  } catch (...) {
    video_capture_.open(device_);
  }

  if (!video_capture_.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open stream: %s", device_.c_str());
    throw std::runtime_error("Cannot open video stream");
  }

  num_frames_ = static_cast<int>(video_capture_.get(cv::CAP_PROP_FRAME_COUNT));
  
  RCLCPP_INFO(this->get_logger(), 
      "Video stream opened: %s, %d frames, %.2f FPS requested",
      device_.c_str(), num_frames_, framerate_);

  // Create publishers with intra-process comms enabled
  raw_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      pub_video_topic_, qos_profile_);
  
  bbox_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      pub_detbox_topic_, qos_profile_);

  // Create subscribers with message_filters for time synchronization
  // Use shared_ptr for automatic memory management and zero-copy
  image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
      this, pub_video_topic_, qos_profile_.get_rmw_qos_profile());
  
  bbox_sub_ = std::make_shared<message_filters::Subscriber<
      vp_interface::msg::BBoxDetList>>(
      this, sub_detbox_topic_, qos_profile_.get_rmw_qos_profile());

  // Create time synchronizer with queue size of 10
  // This synchronizes Image and BBoxDetList messages by timestamp
  time_sync_ = std::make_shared<
      message_filters::TimeSynchronizer<
          sensor_msgs::msg::Image,
          vp_interface::msg::BBoxDetList>>(
      *image_sub_, *bbox_sub_, 10);
  
  time_sync_->registerCallback(
      std::bind(&StreamDemoNode::plot_frame_callback, this,
                std::placeholders::_1, std::placeholders::_2));

  // Create timer for frame publishing
  auto timer_callback = std::bind(&StreamDemoNode::frame_callback, this);
  timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / framerate_),
      timer_callback);

  RCLCPP_INFO(this->get_logger(), 
      "StreamDemo node initialized with intra-process comms enabled");
  RCLCPP_INFO(this->get_logger(), 
      "Publishing raw images to: %s", pub_video_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), 
      "Subscribing to detections from: %s", sub_detbox_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), 
      "Publishing annotated images to: %s", pub_detbox_topic_.c_str());
    RCLCPP_INFO(this->get_logger(),
      "Detector publish resize: min-edge=%d px (upscale=%s), cache=%d frames",
      target_min_edge_px_,
      upscale_to_min_edge_ ? "true" : "false",
      frame_cache_size_);
}

StreamDemoNode::~StreamDemoNode() {
  if (video_capture_.isOpened()) {
    video_capture_.release();
  }
}

void StreamDemoNode::frame_callback() {
  /**
   * Timer callback: Grab frame from video source and publish as Image message.
   * This uses zero-copy with shared_ptr if consumers are using intra-process comms.
   */
  cv::Mat frame;
  bool valid = video_capture_.read(frame);

  frame_num_++;
  if (num_frames_ > 0) {
    frame_num_ %= num_frames_;
  }

  if (!valid || frame_num_ == 0) {
    // Restart video from beginning if no valid frame or end reached
    video_capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
    return;
  }

  const rclcpp::Time stamp = this->get_clock()->now();
  cv::Mat detector_frame = resize_for_detector(frame);

  cache_full_frame(stamp, frame, detector_frame.cols, detector_frame.rows);

  std_msgs::msg::Header header;
  header.stamp = stamp;

  // Publish as unique_ptr to reduce copies in intra-process pipelines.
  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  cv_bridge::CvImage(header, "bgr8", detector_frame).toImageMsg(*msg);

  // Publish - zero-copy if intra-process comms is enabled
  raw_image_pub_->publish(std::move(msg));

    const double progress = (num_frames_ > 0)
      ? (static_cast<double>(frame_num_) * 100.0 / static_cast<double>(num_frames_))
      : 0.0;

  RCLCPP_DEBUG(this->get_logger(),
      "Streaming frame %06d (%dx%d -> %dx%d) from '%s' at %.2f FPS (Processed %.2f%%)",
      frame_num_, frame.cols, frame.rows, detector_frame.cols, detector_frame.rows,
      device_.c_str(), framerate_, progress);
}

void StreamDemoNode::plot_frame_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
    const vp_interface::msg::BBoxDetList::ConstSharedPtr& bbox_msg) {
  /**
   * Synchronization callback: Receive synchronized Image and BBoxDetList messages.
   * Draw bounding boxes on the image and publish annotated result.
   * 
   * Zero-copy benefits:
   * - img_msg is const shared_ptr (no copy needed)
   * - No serialization/deserialization overhead
   * - Direct memory access via cv_bridge
   */
  try {
    CachedFrame cached_frame;
    const rclcpp::Time stamp(img_msg->header.stamp);
    if (!find_cached_frame(stamp, cached_frame)) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "No cached full-resolution frame for stamp_ns=%lld. Skipping bbox overlay.",
          static_cast<long long>(stamp.nanoseconds()));
      return;
    }

    cv::Mat annotated_frame = cached_frame.full_frame.clone();
    const double scale_x = static_cast<double>(cached_frame.full_frame.cols) /
        static_cast<double>(std::max(1, cached_frame.published_width));
    const double scale_y = static_cast<double>(cached_frame.full_frame.rows) /
        static_cast<double>(std::max(1, cached_frame.published_height));

    // Draw boxes in full-resolution coordinates.
    draw_bboxes(annotated_frame, *bbox_msg, scale_x, scale_y);

    // Display locally (optional - remove for headless operation)
    cv::imshow("Annotated Frame", annotated_frame);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {  // 'q' or ESC to quit
      RCLCPP_INFO(this->get_logger(), "Quit requested");
      rclcpp::shutdown();
    }

    // Publish annotated image - also uses zero-copy with shared_ptr
    auto annotated_msg = std::make_unique<sensor_msgs::msg::Image>();
    cv_bridge::CvImage(img_msg->header, "bgr8", annotated_frame).toImageMsg(*annotated_msg);

    bbox_image_pub_->publish(std::move(annotated_msg));

    RCLCPP_DEBUG(this->get_logger(),
        "Processed frame with %zu detections, inference time: %lu us",
        bbox_msg->detections.size(),
        static_cast<unsigned long>(bbox_msg->infertime_us));

  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(),
        "cv_bridge exception: %s", e.what());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(),
        "Error in plot_frame_callback: %s", e.what());
  }
}

void StreamDemoNode::draw_bboxes(
    cv::Mat& frame,
  const vp_interface::msg::BBoxDetList& bbox_msg,
  double scale_x,
  double scale_y) const {
  /**
   * Draw all bounding boxes from BBoxDetList message on the frame.
   * Optimized for performance - uses direct array indexing.
   */
  const auto& detections = bbox_msg.detections;
  const auto& labels = bbox_msg.labels;

  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& det = detections[i];

    // Calculate bounding box coordinates (center + dimensions to corners)
    const double x_center = static_cast<double>(det.box_pos_x) * scale_x;
    const double y_center = static_cast<double>(det.box_pos_y) * scale_y;
    const double box_w = static_cast<double>(det.box_dim_x) * scale_x;
    const double box_h = static_cast<double>(det.box_dim_y) * scale_y;

    int x1 = static_cast<int>(std::lround(x_center - box_w / 2.0));
    int y1 = static_cast<int>(std::lround(y_center - box_h / 2.0));
    int x2 = static_cast<int>(std::lround(x_center + box_w / 2.0));
    int y2 = static_cast<int>(std::lround(y_center + box_h / 2.0));

    // Clamp to image boundaries for safety
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(frame.cols - 1, x2);
    y2 = std::min(frame.rows - 1, y2);

    // Draw rectangle in green (BGR format)
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(0, 255, 0), 2);

    // Prepare label string: "class_id:confidence"
    std::string label;
    if (det.class_id < labels.size()) {
      label = labels[det.class_id];
    } else {
      label = std::to_string(det.class_id);
    }
    label += std::string(":") + std::to_string(det.confidence).substr(0, 4);

    // Draw label background and text
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseline);
    
    cv::rectangle(frame,
                  cv::Point(x1, y1 - text_size.height - baseline - 4),
                  cv::Point(x1 + text_size.width + 4, y1),
                  cv::Scalar(0, 255, 0), -1);  // Filled rectangle
    
    cv::putText(frame, label,
                cv::Point(x1 + 2, y1 - baseline - 2),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 0, 0), 1);  // Black text on green background
  }
}

cv::Mat StreamDemoNode::resize_for_detector(const cv::Mat& frame) const {
  if (frame.empty()) {
    return frame;
  }

  const int min_edge = std::min(frame.cols, frame.rows);
  if (min_edge <= 0) {
    return frame;
  }

  const double target_scale = static_cast<double>(target_min_edge_px_) /
      static_cast<double>(min_edge);
  const bool should_resize = (target_scale < 1.0) || (upscale_to_min_edge_ && target_scale > 1.0);
  if (!should_resize) {
    return frame;
  }

  cv::Mat resized;
  cv::resize(
      frame,
      resized,
      cv::Size(),
      target_scale,
      target_scale,
      target_scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
  return resized;
}

void StreamDemoNode::cache_full_frame(
    const rclcpp::Time& stamp,
    const cv::Mat& full_frame,
    int published_width,
    int published_height) {
  CachedFrame entry;
  entry.stamp = stamp;
  entry.full_frame = full_frame.clone();
  entry.published_width = published_width;
  entry.published_height = published_height;

  std::lock_guard<std::mutex> lock(frame_cache_mutex_);
  frame_cache_.push_back(std::move(entry));
  while (static_cast<int>(frame_cache_.size()) > frame_cache_size_) {
    frame_cache_.pop_front();
  }
}

bool StreamDemoNode::find_cached_frame(
    const rclcpp::Time& stamp,
  CachedFrame& out_frame) {
  constexpr int64_t kToleranceNs = 5 * 1000 * 1000;  // 5 ms fallback tolerance
  std::lock_guard<std::mutex> lock(frame_cache_mutex_);

  if (frame_cache_.empty()) {
    return false;
  }

  for (auto it = frame_cache_.rbegin(); it != frame_cache_.rend(); ++it) {
    if (it->stamp.nanoseconds() == stamp.nanoseconds()) {
      out_frame = *it;
      return true;
    }
  }

  const auto nearest_it = std::min_element(
      frame_cache_.begin(), frame_cache_.end(),
      [&stamp](const CachedFrame& a, const CachedFrame& b) {
        const auto da = std::llabs(a.stamp.nanoseconds() - stamp.nanoseconds());
        const auto db = std::llabs(b.stamp.nanoseconds() - stamp.nanoseconds());
        return da < db;
      });

  if (nearest_it != frame_cache_.end() &&
      std::llabs(nearest_it->stamp.nanoseconds() - stamp.nanoseconds()) <= kToleranceNs) {
    out_frame = *nearest_it;
    return true;
  }

  return false;
}

}  // namespace livestream

// Main entry point
int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);

  try {
    // Create executor with intra-process comms support
    rclcpp::executors::SingleThreadedExecutor executor;
    
    // Create node with intra-process comms enabled in options
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    
    auto node = std::make_shared<livestream::StreamDemoNode>(options);
    executor.add_node(node);
    
    RCLCPP_INFO(node->get_logger(), 
        "StreamDemo C++ node started with zero-copy and intra-process comms");
    
    executor.spin();
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("stream_demo_main"),
        "Exception in StreamDemo: " << e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
