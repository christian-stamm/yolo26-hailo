#include "vp_interface/msg/b_box_det_list.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

// ================================================================
// Pending buffer (indexed round‑robin list)
// ================================================================

struct LetterBox {
    uint64_t                id;
    sensor_msgs::msg::Image image;

    uint original_width;
    uint original_height;
    uint downscale_width;
    uint downscale_height;
};

class PendingBuffer {
  public:
    explicit PendingBuffer(size_t capacity)
        : capacity_(capacity)
    {
    }

    void add(const LetterBox& item)
    {
        pending_.push_back(item);
        auto it         = std::prev(pending_.end());
        index_[item.id] = it;

        if (rr_it_ == pending_.end()) {
            rr_it_ = pending_.begin();
        }
    }

    void get(uint64_t id, LetterBox& item) const
    {
        auto mit = index_.find(id);
        if (mit == index_.end()) {
            throw std::runtime_error("Item not found in pending buffer");
        }

        item = *(mit->second);
    }

    bool remove(uint64_t id)
    {
        auto mit = index_.find(id);
        if (mit == index_.end()) {
            return false;
        }

        auto lit = mit->second;

        // Keep round‑robin iterator valid
        if (rr_it_ == lit) {
            rr_it_ = std::next(rr_it_);
        }

        pending_.erase(lit);
        index_.erase(mit);

        if (pending_.empty()) {
            rr_it_ = pending_.end();
        }
        else if (rr_it_ == pending_.end()) {
            rr_it_ = pending_.begin();
        }

        return true;
    }

    bool full() const
    {
        return pending_.size() >= capacity_;
    }

    bool empty() const
    {
        return pending_.empty();
    }

    const LetterBox& next_rr()
    {
        if (rr_it_ == pending_.end()) {
            rr_it_ = pending_.begin();
        }

        const auto& item = *rr_it_;

        ++rr_it_;
        if (rr_it_ == pending_.end()) {
            rr_it_ = pending_.begin();
        }

        return item;
    }

    size_t size() const
    {
        return pending_.size();
    }

  private:
    size_t                                                       capacity_;
    std::list<LetterBox>                                         pending_;
    std::unordered_map<uint64_t, std::list<LetterBox>::iterator> index_;
    std::list<LetterBox>::iterator                               rr_it_ = pending_.end();
};

// ================================================================
// Evaluation node
// ================================================================

using Name = std::string;
using json = nlohmann::json;
using Path = std::filesystem::path;

class EvaluationNode : public rclcpp::Node {
  public:
    static constexpr const char* UNDEFINED_PARAM = "<NOT SET>";

    EvaluationNode()
        : Node("EvaluationNode")
    {
        const auto dataset_root = Path(declare_parameter<std::string>("dataset_root", UNDEFINED_PARAM));
        const auto dataset_name = Name(declare_parameter<std::string>("dataset_name", UNDEFINED_PARAM));
        const auto label_file   = Path(declare_parameter<std::string>("annotation_file", UNDEFINED_PARAM));
        const auto sub_topic    = declare_parameter<std::string>("sub_det_topic", UNDEFINED_PARAM);
        const auto pub_topic    = declare_parameter<std::string>("pub_img_topic", UNDEFINED_PARAM);
        const auto stream_rate  = declare_parameter<float>("stream_rate", 100.0);
        const auto buffer_size  = declare_parameter<int>("buffer_size", 10);

        Name time        = get_time();
        Name target_file = label_file.stem().string() + "-" + time + label_file.extension().string();

        this->dataset_root = dataset_root / dataset_name;
        annotation_file    = this->dataset_root / "annotations" / label_file;
        prediction_file    = this->dataset_root / "predictions" / target_file;
        evaluation_file    = this->dataset_root / "evaluations" / target_file;

        pendings_ = std::make_unique<PendingBuffer>(buffer_size);

        result_header_ = {
            {"launched", time},
            {"terminated", UNDEFINED_PARAM},
            {"dataset_root", this->dataset_root.string()},
            {"dataset_name", dataset_name},
            {"annotation_file", annotation_file.string()},
            {"prediction_file", prediction_file.string()},
            {"evaluation_file", evaluation_file.string()},
        };

        rclcpp::QoS qos = rclcpp::SensorDataQoS();
        qos.keep_last(1).reliable();

        publisher_ = create_publisher<sensor_msgs::msg::Image>(pub_topic, qos);

        subscription_ = create_subscription<vp_interface::msg::BBoxDetList>(
            sub_topic, qos, std::bind(&EvaluationNode::log_boxes, this, std::placeholders::_1));

        load_dataset();

        timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / stream_rate), std::bind(&EvaluationNode::exec_dataset, this));

        RCLCPP_INFO_STREAM(get_logger(), "Executing evaluation dataset with " << openset.size() << " entries");
    }

  private:
    // ------------------------------------------------------------

    void load_dataset()
    {
        RCLCPP_INFO_STREAM(get_logger(), "Loading dataset from " << annotation_file.string());

        std::ifstream file(annotation_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open dataset file");
        }

        json content;
        file >> content;

        for (const auto& category : content["categories"]) {
            catmapping_.emplace(category["name"].get<std::string>(), category["id"].get<int>());
        }

        for (const auto& entry : content["images"]) {
            openset.emplace(entry["id"].get<uint64_t>(), entry["file_name"].get<Path>());
        }
    }

    // ------------------------------------------------------------

    void exec_dataset()
    {
        if (openset.empty() && pendings_->empty()) {
            safe_predictions();
            rclcpp::shutdown();
            return;
        }

        if (!pendings_->full() && !openset.empty()) {
            const auto [id, file] = *openset.begin();
            openset.erase(id);

            const auto path = this->dataset_root / file;
            if (!std::filesystem::exists(path)) {
                RCLCPP_WARN_STREAM(get_logger(), "Image file does not exist: " << path);
                return;
            }

            cv::Mat img = cv::imread(path.string());
            if (img.empty()) {
                RCLCPP_WARN_STREAM(get_logger(), "Failed to read image " << id);
                return;
            }

            uint original_width  = img.cols;
            uint original_height = img.rows;

            downscale(img);

            uint downscale_width  = img.cols;
            uint downscale_height = img.rows;

            std_msgs::msg::Header header;
            header.stamp    = this->now();
            header.frame_id = std::to_string(id);

            auto msg = sensor_msgs::msg::Image();
            cv_bridge::CvImage(header, "bgr8", img).toImageMsg(msg);

            pendings_->add(LetterBox{
                .id               = id,
                .image            = msg,
                .original_width   = original_width,
                .original_height  = original_height,
                .downscale_width  = downscale_width,
                .downscale_height = downscale_height,
            });
        }

        const auto& item = pendings_->next_rr();
        publisher_->publish(item.image);

        RCLCPP_INFO_STREAM(get_logger(), "Image " << item.id << " sent for eval...");
    }

    // ------------------------------------------------------------

    void log_boxes(const vp_interface::msg::BBoxDetList::ConstSharedPtr msg)
    {
        const uint64_t id = std::stoull(msg->header.frame_id);
        RCLCPP_INFO_STREAM(get_logger(), "Received detections for image " << id);

        if (closedset.contains(id)) {
            return;
        }

        try {

            LetterBox item;
            pendings_->get(id, item);

            json result = nlohmann::ordered_json::array();
            for (const auto& box : msg->detections) {
                int category_id = box.class_id;
                if (box.class_id >= 0 && static_cast<size_t>(box.class_id) < msg->labels.size()) {
                    const auto& category_name = msg->labels[static_cast<size_t>(box.class_id)];
                    const auto  it            = catmapping_.find(category_name);
                    if (it != catmapping_.end()) {
                        category_id = it->second;
                    }
                }

                const float x_min = (box.box_pos_x - box.box_dim_x / 2.0f) * item.original_width /
                                    static_cast<float>(item.downscale_width);
                const float y_min = (box.box_pos_y - box.box_dim_y / 2.0f) * item.original_height /
                                    static_cast<float>(item.downscale_height);
                const float width  = box.box_dim_x * item.original_width / static_cast<float>(item.downscale_width);
                const float height = box.box_dim_y * item.original_height / static_cast<float>(item.downscale_height);

                result.push_back({
                    {"image_id", id},
                    {"category_id", category_id},
                    {"bbox", json::array({x_min, y_min, width, height})},
                    {"score", box.confidence},
                    {"runtime_us", msg->infertime_us},
                });
            }

            pendings_->remove(id);
            closedset.emplace(id, result);
        }
        catch (const std::exception& e) {
            RCLCPP_WARN_STREAM(get_logger(), "Received detections for unknown image " << id);
            return;
        }
    }

    // ------------------------------------------------------------

    void safe_predictions()
    {
        json result         = nlohmann::ordered_json::object();
        result["inference"] = nlohmann::ordered_json::array();

        for (const auto& [_, entries] : closedset) {
            for (const auto& e : entries) {
                result["inference"].push_back(e);
            }
        }

        result["metadata"] = result_header_;
        result["metadata"].emplace("terminated", get_time());

        std::filesystem::create_directories(prediction_file.parent_path());
        std::ofstream out(prediction_file);
        out << result.dump(2);
    }

    // ------------------------------------------------------------

  private:
    static std::string get_time()
    {
        std::time_t t  = std::time(nullptr);
        std::tm     tm = *std::localtime(&t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
        return oss.str();
    }

    static std::string build_filename()
    {
        return "predictions-" + get_time() + ".json";
    }

    // ------------------------------------------------------------

    void downscale(cv::Mat& img)
    {
        constexpr int target_min = 640;
        int           min_side   = std::min(img.cols, img.rows);

        if (min_side <= target_min)
            return;

        double scale = double(target_min) / double(min_side);
        cv::resize(img, img, cv::Size(int(img.cols * scale), int(img.rows * scale)), 0, 0, cv::INTER_AREA);
    }

    Path dataset_root;
    Path annotation_file;
    Path prediction_file;
    Path evaluation_file;
    json result_header_;

    std::map<uint64_t, Path>              openset;
    std::map<uint64_t, std::vector<json>> closedset;
    std::unordered_map<std::string, int>  catmapping_;

    std::unique_ptr<PendingBuffer> pendings_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_;
    rclcpp::Subscription<vp_interface::msg::BBoxDetList>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr                                    timer_;
};

// ================================================================

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EvaluationNode>());
    rclcpp::shutdown();
    return 0;
}
