#include "hailo/backend.hpp"

#include "hailo/hailort.hpp"
#include "hailo/infer_model.hpp"
#include "hailo/vdevice.hpp"
#include "hailo/vstream.hpp"

namespace yolo26::internal {

HailoBackend::HailoBackend()                                   = default;
HailoBackend::~HailoBackend()                                  = default;
HailoBackend::HailoBackend(HailoBackend&&) noexcept            = default;
HailoBackend& HailoBackend::operator=(HailoBackend&&) noexcept = default;

bool HailoBackend::initialize(const std::string& hef_path, std::string& error)
{
    auto vdevice_exp = hailort::VDevice::create();
    if (!vdevice_exp) {
        error = "Failed to create VDevice";
        return false;
    }
    vdevice_ = std::move(vdevice_exp.value());

    auto hef_exp = hailort::Hef::create(hef_path);
    if (!hef_exp) {
        error = "Failed to load HEF from: " + hef_path;
        return false;
    }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = vdevice_->create_configure_params(hef);
    if (!configure_params_exp) {
        error = "Failed to create configure params";
        return false;
    }

    auto network_groups_exp = vdevice_->configure(hef, configure_params_exp.value());
    if (!network_groups_exp || network_groups_exp->empty()) {
        error = "Failed to configure network group";
        return false;
    }
    auto network_groups = std::move(network_groups_exp.value());
    network_group_      = network_groups[0];

    auto input_params = network_group_->make_input_vstream_params(
        false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_params) {
        error = "Failed to create input vstream params";
        return false;
    }

    auto output_params = network_group_->make_output_vstream_params(
        false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_params) {
        error = "Failed to create output vstream params";
        return false;
    }

    auto input_vstreams_exp = hailort::VStreamsBuilder::create_input_vstreams(*network_group_, input_params.value());
    if (!input_vstreams_exp || input_vstreams_exp->empty()) {
        error = "Failed to create input vstreams";
        return false;
    }
    input_vstreams_ = std::move(input_vstreams_exp.value());

    auto output_vstreams_exp = hailort::VStreamsBuilder::create_output_vstreams(*network_group_, output_params.value());
    if (!output_vstreams_exp || output_vstreams_exp->empty()) {
        error = "Failed to create output vstreams";
        return false;
    }
    output_vstreams_ = std::move(output_vstreams_exp.value());

    if (input_vstreams_.size() != 1) {
        error = "Expected a single input stream";
        return false;
    }

    error.clear();
    return true;
}

bool HailoBackend::infer(
    const cv::Mat& model_input_rgb, std::map<std::string, std::vector<float>>& output, std::string& error)
{
    if (!is_ready()) {
        error = "Backend is not initialized";
        return false;
    }

    if (model_input_rgb.empty() || model_input_rgb.type() != CV_8UC3) {
        error = "Model input must be non-empty CV_8UC3 RGB image";
        return false;
    }

    const auto write_status = input_vstreams_[0].write(
        hailort::MemoryView(model_input_rgb.data, model_input_rgb.total() * model_input_rgb.elemSize()));
    if (write_status != HAILO_SUCCESS) {
        error = "Failed writing input tensor";
        return false;
    }

    output.clear();
    for (auto& stream : output_vstreams_) {
        auto& buffer = output[stream.name()];
        buffer.resize(stream.get_frame_size() / sizeof(float));

        const auto read_status = stream.read(hailort::MemoryView(buffer.data(), buffer.size() * sizeof(float)));
        if (read_status != HAILO_SUCCESS) {
            error = "Failed reading output tensor: " + stream.name();
            return false;
        }
    }

    error.clear();
    return true;
}

bool HailoBackend::is_ready() const
{
    return vdevice_ != nullptr && network_group_ != nullptr && !input_vstreams_.empty() && !output_vstreams_.empty();
}

} // namespace yolo26::internal
