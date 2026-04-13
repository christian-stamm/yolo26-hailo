#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace yolo26 {

namespace internal {
    class HailoBackend;
}

struct DetectorConfig {
    std::string hef_path;
    int         input_width          = 640;
    int         input_height         = 640;
    float       confidence_threshold = 0.25f;
};

struct BoundingBox {
    float x        = 0.0f;
    float y        = 0.0f;
    float width    = 0.0f;
    float height   = 0.0f;
    float score    = 0.0f;
    int   class_id = -1;
};

class Detector {
  public:
    explicit Detector(DetectorConfig config);
    ~Detector();

    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    Detector(const Detector&)            = delete;
    Detector& operator=(const Detector&) = delete;

    bool               is_ready() const;
    const std::string& last_error() const;

    std::vector<BoundingBox> infer(const cv::Mat& bgr_image);

  private:
    bool                                    ready_ = false;
    DetectorConfig                          config_;
    std::unique_ptr<internal::HailoBackend> backend_;
    std::string                             last_error_;
};

} // namespace yolo26
