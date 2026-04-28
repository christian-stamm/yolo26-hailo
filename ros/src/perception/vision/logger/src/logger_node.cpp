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

class LoggerNode : public rclcpp::Node {
  public:
    LoggerNode()
        : rclcpp::Node("LoggerNode")
    {
        // const auto hef_path             = declare_parameter<std::string>("hef_path", "<NO HEF PATH>");
        // const auto image_topic          = declare_parameter<std::string>("image_topic", "/image_raw");
        // const auto bboxdet_topic        = declare_parameter<std::string>("bboxdet_topic", "/yolo26/detstream");
        // const auto bboxdet_service      = declare_parameter<std::string>("bboxdet_service", "/yolo26/detservice");
        // const auto input_width          = declare_parameter<int>("input_width", 640);
        // const auto input_height         = declare_parameter<int>("input_height", 640);
        // const auto confidence_threshold = declare_parameter<double>("confidence_threshold", 0.25);
        // const auto queue_size           = declare_parameter<int>("queue_size", 10);

        // DetectorConfig detector_config;
        // detector_config.hef_path             = hef_path;
        // detector_config.input_width          = input_width;
        // detector_config.input_height         = input_height;
        // detector_config.confidence_threshold = static_cast<float>(confidence_threshold);

        // detector_ = std::make_unique<Detector>(std::move(detector_config));
        // if (!detector_->is_ready()) {
        //     throw std::runtime_error("Detector initialization failed: " + detector_->last_error());
        // }

        // auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

        // publisher_ = create_publisher<vp_interface::msg::BBoxDetList>(bboxdet_topic, qos);

        // subscription_ = create_subscription<sensor_msgs::msg::Image>(
        //     image_topic, qos,                                         //
        //     [this](                                                   //
        //         const sensor_msgs::msg::Image::ConstSharedPtr& msg) { //
        //         this->topic_cb(msg);
        //     });

        // service_ = create_service<vp_interface::srv::BBoxDet>(
        //     bboxdet_service,                                                                       //
        //     [this](                                                                                //
        //         const std::shared_ptr<vp_interface::srv::BBoxDet::Request> request, //
        //         std::shared_ptr<vp_interface::srv::BBoxDet::Response>      response) {   //
        //         this->service_cb(request, response);
        //     });

        // RCLCPP_INFO(
        //     get_logger(), "Started yolo26 ROS node. image_topic=%s bboxdet_topic=%s bboxdet_service=%s",
        //     image_topic.c_str(), bboxdet_topic.c_str(), bboxdet_service.c_str());
    }

  private:
    // bool process_image(
    //     const sensor_msgs::msg::Image& image_msg, vp_interface::msg::BBoxDetList& bbox_msg)
    // {
    //     cv_bridge::CvImagePtr cv_ptr;
    //     try {
    //         cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    //     }
    //     catch (const cv_bridge::Exception& ex) {
    //         RCLCPP_WARN(get_logger(), "Failed to convert image: %s", ex.what());
    //         return false;
    //     }

    //     const auto infer_start = std::chrono::steady_clock::now();
    //     const auto detections  = detector_->infer(cv_ptr->image);
    //     const auto infer_end   = std::chrono::steady_clock::now();

    //     if (!detector_->last_error().empty()) {
    //         RCLCPP_WARN(get_logger(), "Inference failed: %s", detector_->last_error().c_str());
    //         return false;
    //     }

    //     const auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count();

    //     bbox_msg.header       = image_msg.header;
    //     bbox_msg.infertime_us = static_cast<uint64_t>(infer_us);
    //     bbox_msg.detections.reserve(detections.size());

    //     for (const auto& box : detections) {
    //         vp_interface::msg::BBoxDet det;
    //         det.class_id   = box.class_id;
    //         det.confidence = box.score;
    //         det.box_pos_x  = box.x;
    //         det.box_pos_y  = box.y;
    //         det.box_dim_x  = box.width;
    //         det.box_dim_y  = box.height;
    //         bbox_msg.detections.push_back(std::move(det));
    //     }

    //     std::sort(
    //         bbox_msg.detections.begin(), bbox_msg.detections.end(),
    //         [](const vp_interface::msg::BBoxDet& lhs,
    //            const vp_interface::msg::BBoxDet& rhs) { return lhs.confidence < rhs.confidence; });

    //     RCLCPP_INFO(
    //         get_logger(), "Inference took %zu us and %zu Object(s) were found.", //
    //         infer_us, bbox_msg.detections.size());

    //     return true;
    // }

    // void topic_cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    // {
    //     vp_interface::msg::BBoxDetList bbox_msg;

    //     if (!process_image(*msg, bbox_msg)) {
    //         return;
    //     }

    //     publisher_->publish(std::move(bbox_msg));
    // }

    // void service_cb(
    //     const std::shared_ptr<vp_interface::srv::BBoxDet::Request> request,
    //     std::shared_ptr<vp_interface::srv::BBoxDet::Response>      response)
    // {
    //     process_image(request->sample, response->boxes);
    // }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr                    subscription_;
    rclcpp::Publisher<vp_interface::msg::BBoxDetList>::SharedPtr publisher_;
    rclcpp::Service<vp_interface::srv::BBoxDet>::SharedPtr       service_;
};



int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LoggerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
