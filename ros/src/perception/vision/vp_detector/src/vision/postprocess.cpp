#include "vision/postprocess.hpp"

#include <array>
#include <cmath>

namespace yolo26::internal {

namespace {

    float sigmoid(const float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    bool map_output_tensors(
        const std::map<std::string, std::vector<float>>& output_buffers, std::array<const float*, 3>& cls,
        std::array<const float*, 3>& reg)
    {
        cls.fill(nullptr);
        reg.fill(nullptr);

        for (const auto& [name, values] : output_buffers) {
            (void)name;
            const size_t count = values.size();

            if (count == 512000) {
                cls[0] = values.data();
            }
            else if (count == 128000) {
                cls[1] = values.data();
            }
            else if (count == 32000) {
                cls[2] = values.data();
            }
            else if (count == 25600) {
                reg[0] = values.data();
            }
            else if (count == 6400) {
                reg[1] = values.data();
            }
            else if (count == 1600) {
                reg[2] = values.data();
            }
        }

        return cls[0] && cls[1] && cls[2] && reg[0] && reg[1] && reg[2];
    }

} // namespace

const std::vector<std::string>& coco_class_names()
{
    static const std::vector<std::string> classes = {"person",        "bicycle",      "car",
                                                     "motorcycle",    "airplane",     "bus",
                                                     "train",         "truck",        "boat",
                                                     "traffic light", "fire hydrant", "stop sign",
                                                     "parking meter", "bench",        "bird",
                                                     "cat",           "dog",          "horse",
                                                     "sheep",         "cow",          "elephant",
                                                     "bear",          "zebra",        "giraffe",
                                                     "backpack",      "umbrella",     "handbag",
                                                     "tie",           "suitcase",     "frisbee",
                                                     "skis",          "snowboard",    "sports ball",
                                                     "kite",          "baseball bat", "baseball glove",
                                                     "skateboard",    "surfboard",    "tennis racket",
                                                     "bottle",        "wine glass",   "cup",
                                                     "fork",          "knife",        "spoon",
                                                     "bowl",          "banana",       "apple",
                                                     "sandwich",      "orange",       "broccoli",
                                                     "carrot",        "hot dog",      "pizza",
                                                     "donut",         "cake",         "chair",
                                                     "couch",         "potted plant", "bed",
                                                     "dining table",  "toilet",       "tv",
                                                     "laptop",        "mouse",        "remote",
                                                     "keyboard",      "cell phone",   "microwave",
                                                     "oven",          "toaster",      "sink",
                                                     "refrigerator",  "book",         "clock",
                                                     "vase",          "scissors",     "teddy bear",
                                                     "hair drier",    "toothbrush"};
    return classes;
}

std::vector<ModelDetection> decode_outputs(
    const std::map<std::string, std::vector<float>>& output_buffers, float confidence_threshold)
{
    std::array<const float*, 3> cls = {};
    std::array<const float*, 3> reg = {};
    if (!map_output_tensors(output_buffers, cls, reg)) {
        return {};
    }

    if (confidence_threshold <= 0.0f) {
        confidence_threshold = 0.001f;
    }
    if (confidence_threshold >= 1.0f) {
        confidence_threshold = 0.999f;
    }
    const float logit_threshold = -std::log(1.0f / confidence_threshold - 1.0f);

    constexpr std::array<int, 3> kStrides = {8, 16, 32};
    constexpr std::array<int, 3> kGrids   = {80, 40, 20};

    std::vector<ModelDetection> detections;

    for (size_t scale = 0; scale < kStrides.size(); ++scale) {
        const int    grid     = kGrids[scale];
        const int    anchors  = grid * grid;
        const float* cls_data = cls[scale];
        const float* reg_data = reg[scale];

        for (int i = 0; i < anchors; ++i) {
            float     max_logit  = -1000.0f;
            int       best_class = -1;
            const int cls_offset = i * 80;

            for (int c = 0; c < 80; ++c) {
                const float logit = cls_data[cls_offset + c];
                if (logit > max_logit) {
                    max_logit  = logit;
                    best_class = c;
                }
            }

            if (max_logit <= logit_threshold || best_class < 0) {
                continue;
            }

            const int   row    = i / grid;
            const int   col    = i % grid;
            const float stride = static_cast<float>(kStrides[scale]);

            const int   reg_offset = i * 4;
            const float l          = reg_data[reg_offset + 0];
            const float t          = reg_data[reg_offset + 1];
            const float r          = reg_data[reg_offset + 2];
            const float b          = reg_data[reg_offset + 3];

            ModelDetection det;
            det.x1       = (static_cast<float>(col) + 0.5f - l) * stride;
            det.y1       = (static_cast<float>(row) + 0.5f - t) * stride;
            det.x2       = (static_cast<float>(col) + 0.5f + r) * stride;
            det.y2       = (static_cast<float>(row) + 0.5f + b) * stride;
            det.score    = sigmoid(max_logit);
            det.class_id = best_class;
            detections.push_back(det);
        }
    }

    return detections;
}

} // namespace yolo26::internal
