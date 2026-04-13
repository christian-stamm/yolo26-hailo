# yolo26-hailo

This repository now follows a split architecture:

- Core inference (`yolo26`) is a standalone CMake library and demo binary.
- ROS 2 integration lives in a dedicated ROS 2 workspace at `ros2_ws/`.

ROS 2 is only a wrapper around the standalone inference API.

## 1) Build standalone inference (no ROS required)

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the demo:

```bash
./build/src/target/inference/yolo26_demo \
    /workspaces/yolo26-hailo/res/models/model.hef \
    /workspaces/yolo26-hailo/res/sample.jpg \
    0.5
```

## 2) Install core library for ROS wrapper consumption

Install `yolo26` into a local prefix so ROS 2 package can `find_package(yolo26 CONFIG)`:

```bash
cmake --install build --prefix /workspaces/yolo26-hailo/install/core
```

## 3) Build ROS 2 wrapper workspace

## 4) Run ROS 2 node

Using launch file:

```bash
ros2 launch visual_perception detector.launch.py \
    hef_path:=/workspaces/yolo26-hailo/res/models/model.hef
```

Or directly:

```bash
ros2 run visual_perception yolo26_detector_node --ros-args \
    -p hef_path:=/workspaces/yolo26-hailo/res/models/model.hef \
    -p image_topic:=/image_raw \
    -p detections_topic:=/yolo26/detections
```

## 5) Export HEF file (optional)

```bash
python3 -m src.server.export.cli \
    --variant yolo26n \
    --target hailo8 \
    --onnx /workspaces/yolo26-hailo/res/models/yolo26n.onnx \
    --calib_dir /workspaces/yolo26-hailo/res/datasets/calib_images \
    --tag stadtup_coco
```


Generate Hailo HEF File python3 -m src.host.export.cli --variant yolo26n --target hailo8 --onnx /workspaces/yolo26-hailo/res/models/yolo26n.onnx --calib_dir /workspaces/yolo26-hailo/res/datasets/calib_images --tag stadtup_coco

/workspaces/yolo26-hailo/build/src/node/infer/yolo26_demo /workspaces/yolo26-hailo/res/models/model.hef /workspaces/yolo26-hailo/res/sample.jpg 0.5

sudo mkdir -p /usr/local/share/ca-certificates/zf sudo cp /tmp/ZF_.crt /usr/local/share/ca-certificates/zf/ sudo chmod 644 /usr/local/share/ca-certificates/zf/.crt sudo update-ca-certificates

xhost +local:docker
sudo nano /etc/dhcpcd.conf