#include "detector_api.hpp"

#include "hailo/backend.hpp"
#include "vision/postprocess.hpp"
#include "vision/preprocess.hpp"

#include <algorithm>
#include <exception>
#include <map>
#include <utility>

namespace yolo26 {

Detector::Detector(DetectorConfig config)
    : config_(std::move(config))
    , backend_(std::make_unique<internal::HailoBackend>())
{
    if (config_.hef_path.empty()) {
        last_error_ = "DetectorConfig.hef_path must not be empty";
        return;
    }
    if (config_.input_width <= 0 || config_.input_height <= 0) {
        last_error_ = "Detector input dimensions must be positive";
        return;
    }

    if (!backend_ || !backend_->initialize(config_.hef_path, last_error_)) {
        return;
    }

    ready_ = true;
    last_error_.clear();
}

Detector::~Detector()                              = default;
Detector::Detector(Detector&&) noexcept            = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

bool Detector::is_ready() const
{
    return ready_;
}

const std::string& Detector::last_error() const
{
    return last_error_;
}

std::vector<BoundingBox> Detector::infer(const cv::Mat& bgr_image)
{
    if (!ready_ || !backend_) {
        return {};
    }

    try {
        const auto prep = internal::letterbox_to_model(bgr_image, config_.input_width, config_.input_height);

        std::map<std::string, std::vector<float>> outputs;
        if (!backend_->infer(prep.rgb_image, outputs, last_error_)) {
            return {};
        }

        auto raw = internal::decode_outputs(outputs, config_.confidence_threshold);

        const auto&              class_names = internal::coco_class_names();
        std::vector<BoundingBox> boxes;
        boxes.reserve(raw.size());

        for (const auto& det : raw) {
            float x1 = (det.x1 - static_cast<float>(prep.pad_w)) / prep.scale;
            float y1 = (det.y1 - static_cast<float>(prep.pad_h)) / prep.scale;
            float x2 = (det.x2 - static_cast<float>(prep.pad_w)) / prep.scale;
            float y2 = (det.y2 - static_cast<float>(prep.pad_h)) / prep.scale;

            x1 = std::clamp(x1, 0.0f, static_cast<float>(prep.original_w));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(prep.original_h));
            x2 = std::clamp(x2, 0.0f, static_cast<float>(prep.original_w));
            y2 = std::clamp(y2, 0.0f, static_cast<float>(prep.original_h));

            const float width  = std::max(0.0f, x2 - x1);
            const float height = std::max(0.0f, y2 - y1);

            if (width <= 0.0f || height <= 0.0f) {
                continue;
            }

            BoundingBox box;
            box.x        = x1 + width / 2.0f; // Convert to center-based coordinates
            box.y        = y1 + height / 2.0f; // Convert to center-based coordinates
            box.width    = width;
            box.height   = height;
            box.score    = det.score;
            box.class_id = det.class_id;
            boxes.push_back(std::move(box));
        }

        last_error_.clear();
        return boxes;
    }
    catch (const std::exception& ex) {
        last_error_ = ex.what();
        return {};
    }
}

} // namespace yolo26
