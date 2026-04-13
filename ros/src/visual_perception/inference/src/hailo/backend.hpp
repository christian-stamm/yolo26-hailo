#pragma once

#include "hailo/hailort.hpp"
#include "hailo/vstream.hpp"

#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace yolo26::internal {

class HailoBackend {
  public:
    HailoBackend();
    ~HailoBackend();

    HailoBackend(const HailoBackend&)            = delete;
    HailoBackend& operator=(const HailoBackend&) = delete;

    HailoBackend(HailoBackend&&) noexcept;
    HailoBackend& operator=(HailoBackend&&) noexcept;

    bool initialize(const std::string& hef_path, std::string& error);
    bool infer(const cv::Mat& model_input_rgb, std::map<std::string, std::vector<float>>& output, std::string& error);
    bool is_ready() const;

  private:
    std::unique_ptr<hailort::VDevice>                vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::vector<hailort::InputVStream>               input_vstreams_;
    std::vector<hailort::OutputVStream>              output_vstreams_;
};

} // namespace yolo26::internal
