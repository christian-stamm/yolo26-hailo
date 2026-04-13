#include "vision/preprocess.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace yolo26::internal {

LetterboxInfo letterbox_to_model(const cv::Mat& bgr_image, int target_width, int target_height)
{
    if (bgr_image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    if (target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument("Target dimensions must be positive");
    }

    const int src_h = bgr_image.rows;
    const int src_w = bgr_image.cols;

    const float scale = std::min(
        static_cast<float>(target_width) / static_cast<float>(src_w),
        static_cast<float>(target_height) / static_cast<float>(src_h));

    const int resized_w = static_cast<int>(src_w * scale);
    const int resized_h = static_cast<int>(src_h * scale);

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(resized_w, resized_h));

    const int pad_w = (target_width - resized_w) / 2;
    const int pad_h = (target_height - resized_h) / 2;

    cv::Mat padded(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, resized_w, resized_h)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    LetterboxInfo info;
    info.rgb_image  = std::move(rgb);
    info.scale      = scale;
    info.pad_w      = pad_w;
    info.pad_h      = pad_h;
    info.original_w = src_w;
    info.original_h = src_h;
    return info;
}

} // namespace yolo26::internal
