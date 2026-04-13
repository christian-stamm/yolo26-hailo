#include "vp_interface/msg/b_box_det_list.hpp"

#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/publisher_base.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

class VisuNode : public rclcpp::Node {
  public:
    VisuNode()
        : Node("VisuNode")
    {
        const auto video_topic = declare_parameter<std::string>("video_topic", "detector/fast/img");
        const auto bbox_topic  = declare_parameter<std::string>("bbox_topic", "detector/fast/det");
        const auto buffer_size = declare_parameter<int>("buffer_size", 10);

        display_window_  = declare_parameter<bool>("display_window", true);
        publish_results_ = declare_parameter<bool>("publish_results", false);

        rclcpp::QoS qos = rclcpp::SensorDataQoS();
        qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        qos.best_effort();
        qos.keep_last(1);

        if (publish_results_) {
            image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("stream", qos);
        }

        image_sub_.subscribe(this, video_topic, qos.get_rmw_qos_profile());
        bbox_sub_.subscribe(this, bbox_topic, qos.get_rmw_qos_profile());

        time_sync_ = std::make_shared<
            message_filters::TimeSynchronizer<sensor_msgs::msg::Image, vp_interface::msg::BBoxDetList>>(
            image_sub_, bbox_sub_, buffer_size);

        time_sync_->registerCallback(
            std::bind(&VisuNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "VisuNode listening on '%s' and '%s'", video_topic.c_str(), bbox_topic.c_str());
    }

    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr&        image_msg,
        const vp_interface::msg::BBoxDetList::ConstSharedPtr& bbox_msg)
    {
        RCLCPP_INFO(
            get_logger(), "Received synchronized image and bounding box messages with timestamp %u.%u", //
            image_msg->header.stamp.sec, image_msg->header.stamp.nanosec);

        // Convert ROS image message to OpenCV format
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Draw bounding boxes on the image
        for (const auto& det : bbox_msg->detections) {
            cv::Rect box(det.box_pos_x, det.box_pos_y, det.box_dim_x, det.box_dim_y);
            cv::rectangle(cv_ptr->image, box, cv::Scalar(0, 255, 0), 2);
            std::string label     = "ID: " + std::to_string(det.class_id) + " Conf: " + std::to_string(det.confidence);
            int         baseline  = 0;
            cv::Size    text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(
                cv_ptr->image, cv::Point(box.x, box.y - text_size.height - baseline),
                cv::Point(box.x + text_size.width, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(
                cv_ptr->image, label, cv::Point(box.x, box.y - baseline), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1);
        }

        // Optionally Display the image with bounding boxes
        if (display_window_) {
            cv::imshow("Detection Visualization", cv_ptr->image);
            cv::waitKey(1); // Needed to update the OpenCV window
        }

        // Optionally publish results
        if (publish_results_ && image_pub_) {
            image_pub_->publish(*cv_ptr->toImageMsg());
        }
    }

    ~VisuNode() override {}

  private:
    bool display_window_;
    bool publish_results_;

    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_pub_;
    message_filters::Subscriber<sensor_msgs::msg::Image>        image_sub_;
    message_filters::Subscriber<vp_interface::msg::BBoxDetList> bbox_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, vp_interface::msg::BBoxDetList>>
        time_sync_;
};

/* ---------- main ---------- */

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisuNode>());
    rclcpp::shutdown();
    return 0;
}
