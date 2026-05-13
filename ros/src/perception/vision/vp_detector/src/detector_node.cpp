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
        if (prefix == "/") {
            prefix.clear();
        }
        if (!prefix.empty() && prefix.back() != '/') {
            prefix += '/';
        }

        const std::string img_topic = prefix + "detector/img";
        const std::string det_topic = prefix + "detector/det";

        RCLCPP_INFO(get_logger(), "Started yolo26 ROS node in namespace '%s'", prefix.c_str());
        RCLCPP_INFO(get_logger(), "Using HEF Path: %s", hef_path.c_str());
        RCLCPP_INFO(
            get_logger(), "Receiving Images on '%s' and publishing detections on '%s'", //
            img_topic.c_str(), det_topic.c_str()                                        //
        );

        detector_ = std::make_unique<Detector>(std::move(detector_config));
        if (!detector_->is_ready()) {
            throw std::runtime_error("Detector initialization failed: " + detector_->last_error());
        }

        rclcpp::QoS qos = rclcpp::SensorDataQoS();
        qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        qos.best_effort();
        qos.keep_last(1);

        publisher_ = create_publisher<vp_interface::msg::BBoxDetList>(det_topic, qos);

        subscription_ = create_subscription<sensor_msgs::msg::Image>(
            img_topic, qos, std::bind(&DetectorRosNode::on_stream, this, std::placeholders::_1));

        infer_thread_ = std::thread(&DetectorRosNode::infer_daemon, this);

        RCLCPP_INFO(get_logger(), "Node is initialized. Waiting for images...");
    }

    ~DetectorRosNode() override
    {
        if (rclcpp::ok()) {
            RCLCPP_INFO(get_logger(), "Shutting down yolo26 ROS node...");
            rclcpp::shutdown();
        }

        if (infer_thread_.joinable()) {
            infer_thread_.join();
        }
    }

  private:
    void on_stream(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        {
            std::lock_guard<std::mutex> lock(stream_mtx_);
            stream_msg_ = msg;
        }
        stream_notifier_.notify_one();
    }

    void infer_daemon()
    {

        while (rclcpp::ok()) {
            sensor_msgs::msg::Image::ConstSharedPtr local_msg;
            vp_interface::msg::BBoxDetList          bbox_msg;

            {
                std::unique_lock<std::mutex> lock(stream_mtx_);
                stream_notifier_.wait(lock, [&] { return stream_msg_ || !rclcpp::ok(); });

                local_msg = stream_msg_;
                stream_msg_.reset();
            }

            auto cv_ptr = cv_bridge::toCvShare(local_msg, sensor_msgs::image_encodings::BGR8);

            if (!cv_ptr || cv_ptr->image.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            cv::Mat img = cv_ptr->image.clone();

            auto start      = std::chrono::steady_clock::now();
            auto detections = detector_->infer(img);
            auto end        = std::chrono::steady_clock::now();

            if (!detector_->last_error().empty()) {
                RCLCPP_WARN(get_logger(), "%s", detector_->last_error().c_str());
                return;
            }

            const auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            bbox_msg.header       = local_msg->header;
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
                get_logger(), "Inference took %ld us and %zu Object(s) were found.", infer_us,
                bbox_msg.detections.size());

            publisher_->publish(bbox_msg);
        }
    }

    std::thread                                                  infer_thread_;
    std::mutex                                                   stream_mtx_;
    sensor_msgs::msg::Image::ConstSharedPtr                      stream_msg_;
    std::condition_variable                                      stream_notifier_;
    std::unique_ptr<Detector>                                    detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr     subscription_;
    rclcpp::Publisher<vp_interface::msg::BBoxDetList>::SharedPtr publisher_;
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
