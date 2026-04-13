#include "vp_interface/msg/b_box_det_list.hpp"

#include <chrono>
#include <cstdint>
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

class CVStreamNode : public rclcpp::Node {
  public:
    CVStreamNode()
        : Node("CVStreamNode")
        , frame_ID(0)
    {
        std::string pub_video_topic  = declare_parameter<std::string>("pub_video_topic", "detector/fast/img");
        std::string pub_video_source = declare_parameter<std::string>("pub_video_source", "0");
        float       pub_video_rate   = declare_parameter<float>("pub_video_rate", 30.0);

        rclcpp::QoS qos = rclcpp::SensorDataQoS();
        qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        qos.best_effort();
        qos.keep_last(1);

        try {
            int device_id = std::stoi(pub_video_source);
            video_capture_.open(device_id);
        }
        catch (...) {
            video_capture_.open(pub_video_source);
        }

        if (!video_capture_.isOpened()) {
            RCLCPP_ERROR(get_logger(), "CVStreamNode could not load video stream '%s'", pub_video_source.c_str());
        }

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(pub_video_topic, qos);

        // Create timer for frame publishing
        auto timer_callback = [this]() { frame_callback(); };
        timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / pub_video_rate), timer_callback);

        RCLCPP_INFO(
            get_logger(), "CVStreamNode streaming from '%s' and publishing to '%s'", pub_video_source.c_str(),
            pub_video_topic.c_str());
    }

    void frame_callback(int retry = 0)
    {
        cv::Mat frame;
        bool    valid = video_capture_.read(frame);

        if (!valid) {

            RCLCPP_WARN(get_logger(), "CVStreamNode failed loading next frame");

            if (3 < retry) {
                RCLCPP_ERROR(get_logger(), "CVStreamNode max retries reached.");
                return;
            }

            // Restart video from beginning if no valid frame or end reached
            RCLCPP_INFO(get_logger(), "CVStreamNode restarting stream...");
            video_capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
            frame_callback(++retry);
            return;
        }

        const rclcpp::Time    stamp = this->get_clock()->now();
        std_msgs::msg::Header header;
        header.stamp    = stamp;
        header.frame_id = std::to_string(frame_ID++);

        auto msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(header, "bgr8", frame).toImageMsg(*msg);

        // Publish - zero-copy if intra-process comms is enabled
        publisher_->publish(std::move(msg));

        RCLCPP_INFO_STREAM(get_logger(), "Frame published at " << stamp.nanoseconds() << " ns.");
    }

    ~CVStreamNode() override {}

  private:
    uint64_t                                              frame_ID;
    cv::VideoCapture                                      video_capture_;
    rclcpp::TimerBase::SharedPtr                          timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

/* ---------- main ---------- */

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CVStreamNode>());
    rclcpp::shutdown();
    return 0;
}
