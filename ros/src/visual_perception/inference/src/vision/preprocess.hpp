#pragma once

#include <opencv2/core.hpp>

namespace yolo26::internal {

struct LetterboxInfo {
    cv::Mat rgb_image;
    float   scale      = 1.0f;
    int     pad_w      = 0;
    int     pad_h      = 0;
    int     original_w = 0;
    int     original_h = 0;
};

LetterboxInfo letterbox_to_model(const cv::Mat& bgr_image, int target_width, int target_height);

} // namespace yolo26::internal
