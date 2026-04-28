#pragma once

#include <map>
#include <string>
#include <vector>

namespace yolo26::internal {

struct ModelDetection {
    float x1       = 0.0f;
    float y1       = 0.0f;
    float x2       = 0.0f;
    float y2       = 0.0f;
    float score    = 0.0f;
    int   class_id = -1;
};

const std::vector<std::string>& coco_class_names();
std::vector<ModelDetection>     decode_outputs(
        const std::map<std::string, std::vector<float>>& output_buffers, float confidence_threshold);

} // namespace yolo26::internal
