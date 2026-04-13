#include "cv_bridge/cv_bridge.h"
#include "detector_api.hpp"
#include "vp_interface/msg/b_box_det_list.hpp"
#include "vp_interface/srv/b_box_det.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace yolo26 {

class DetectorRosNode : public rclcpp::Node {
  public:
    DetectorRosNode()
        : rclcpp::Node("yolo26_detector_node")
    {
        const auto hef_path    = declare_parameter<std::string>("hef_path", "<NO HEF PATH>");
        const auto conf_thresh = declare_parameter<double>("conf_thresh", 0.25);
        const auto img_width   = declare_parameter<int>("img_width", 640);
        const auto img_height  = declare_parameter<int>("img_height", 640);

        DetectorConfig detector_config;
        detector_config.hef_path             = hef_path;
        detector_config.input_width          = img_width;
        detector_config.input_height         = img_height;
        detector_config.confidence_threshold = conf_thresh;

        std::string prefix = this->get_namespace();
        if (prefix.empty() || prefix.back() != '/') {
            prefix += '/';
        }

        const std::string safe_img_topic = prefix + "detector/safe/img";
        const std::string fast_img_topic = prefix + "detector/fast/img";
        const std::string safe_det_topic = prefix + "detector/safe/det";
        const std::string fast_det_topic = prefix + "detector/fast/det";

        RCLCPP_INFO(get_logger(), "Started yolo26 ROS node in namespace '%s'", prefix.c_str());
        RCLCPP_INFO(get_logger(), "Using HEF Path: %s", hef_path.c_str());
        RCLCPP_INFO(
            get_logger(), "Receiving Images on '%s' and '%s' and publishing detections on '%s' and '%s'", //
            safe_img_topic.c_str(), fast_img_topic.c_str(), safe_det_topic.c_str(), fast_det_topic.c_str());

        detector_ = std::make_unique<Detector>(std::move(detector_config));
        if (!detector_->is_ready()) {
            throw std::runtime_error("Detector initialization failed: " + detector_->last_error());
        }

        rclcpp::QoS safe_qos = rclcpp::SensorDataQoS();
        safe_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        safe_qos.reliable();
        safe_qos.keep_last(1);

        rclcpp::QoS fast_qos = rclcpp::SensorDataQoS();
        fast_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        fast_qos.best_effort();
        fast_qos.keep_last(1);

        safe_publisher_ = create_publisher<vp_interface::msg::BBoxDetList>(safe_det_topic, safe_qos);
        fast_publisher_ = create_publisher<vp_interface::msg::BBoxDetList>(fast_det_topic, fast_qos);

        safe_subscription_ = create_subscription<sensor_msgs::msg::Image>(
            safe_img_topic, safe_qos, std::bind(&DetectorRosNode::on_safe_stream, this, std::placeholders::_1));
        fast_subscription_ = create_subscription<sensor_msgs::msg::Image>(
            fast_img_topic, fast_qos, std::bind(&DetectorRosNode::on_fast_stream, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Node is initialized. Waiting for images...");
    }

  private:
    void process_image(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg, vp_interface::msg::BBoxDetList& bbox_msg)
    {
        cv_bridge::CvImagePtr cv_ptr;

        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (const cv_bridge::Exception& ex) {
            RCLCPP_WARN(get_logger(), "Failed to convert image: %s", ex.what());
            return;
        }

        const auto infer_start = std::chrono::steady_clock::now();
        const auto detections  = detector_->infer(cv_ptr->image);
        const auto infer_end   = std::chrono::steady_clock::now();

        if (!detector_->last_error().empty()) {
            RCLCPP_WARN(get_logger(), "Inference failed: %s", detector_->last_error().c_str());
            return;
        }

        const auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count();

        bbox_msg.header       = image_msg->header;
        bbox_msg.labels       = detector_->get_classes();
        bbox_msg.infertime_us = static_cast<uint64_t>(infer_us);
        bbox_msg.detections.reserve(detections.size());

        for (const auto& box : detections) {
            vp_interface::msg::BBoxDet det;
            det.class_id   = box.class_id;
            det.confidence = box.score;
            det.box_pos_x  = box.x;
            det.box_pos_y  = box.y;
            det.box_dim_x  = box.width;
            det.box_dim_y  = box.height;
            bbox_msg.detections.push_back(std::move(det));
        }

        std::sort(
            bbox_msg.detections.begin(), bbox_msg.detections.end(),
            [](const vp_interface::msg::BBoxDet& lhs, const vp_interface::msg::BBoxDet& rhs) {
                return lhs.confidence > rhs.confidence;
            });

        RCLCPP_INFO(
            get_logger(), "Inference took %zu us and %zu Object(s) were found.", //
            infer_us, bbox_msg.detections.size());
    }

    void on_fast_stream(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg)
    {
        vp_interface::msg::BBoxDetList bbox_msg;
        process_image(image_msg, bbox_msg);
        fast_publisher_->publish(bbox_msg);
    }

    void on_safe_stream(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg)
    {
        vp_interface::msg::BBoxDetList bbox_msg;
        process_image(image_msg, bbox_msg);
        safe_publisher_->publish(bbox_msg);
    }

    std::unique_ptr<Detector>                                    detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr     safe_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr     fast_subscription_;
    rclcpp::Publisher<vp_interface::msg::BBoxDetList>::SharedPtr safe_publisher_;
    rclcpp::Publisher<vp_interface::msg::BBoxDetList>::SharedPtr fast_publisher_;
};

} // namespace yolo26

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<yolo26::DetectorRosNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
