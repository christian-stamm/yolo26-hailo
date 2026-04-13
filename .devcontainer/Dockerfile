FROM global-registry.cloud.zf-world.com/docker.io/library/ubuntu:jammy
ARG HWARCH=amd64
ARG HAILOVERSION=4.23.0
ARG UBUNTU_CODENAME=jammy
ARG ROS_APT_SOURCE_VERSION=1.2.0

ENV GIT_SSL_NO_VERIFY=1
ENV DEBIAN_FRONTEND=noninteractive

# Update System
RUN apt update && apt upgrade -y

# Install DEPS
ARG SYS_DEPS="gnupg git curl wget ca-certificates software-properties-common net-tools locales iputils-ping"
ARG CPP_DEPS="build-essential gcc cmake pkg-config clangd"
ARG PYTHON_DEPS="python3 python3-venv python3-pip python3-dev python3-setuptools python3-tk"
ARG LIB_DEPS="graphviz graphviz-dev libopencv-dev xtensor-dev nlohmann-json3-dev"
RUN apt update && apt install -y $SYS_DEPS $CPP_DEPS $PYTHON_DEPS $LIB_DEPS
RUN add-apt-repository universe
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install --system

# Install ROS
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG="en_US.UTF-8"
RUN curl -L -o /tmp/ros2-apt-source.deb https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME}_all.deb
RUN dpkg -i /tmp/ros2-apt-source.deb
RUN apt update && apt upgrade -y
RUN apt install -y ros-humble-ros-base ros-dev-tools python3-colcon-common-extensions ros-humble-cv-bridge ros-humble-rqt-image-view ros-humble-v4l2-camera ros-humble-vision-msgs ros-humble-rviz2 ros-humble-spinnaker-camera-driver
RUN echo 'source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc

# Install Runtime
WORKDIR /tmp
COPY ./hailo*.deb .
COPY ./hailo*.whl .
RUN mkdir -p /opt/hailo && dpkg-deb -x /tmp/hailort_${HAILOVERSION}_${HWARCH}.deb /opt/hailo
RUN ln -s /opt/hailo/usr/bin/hailortcli /usr/bin/hailortcli && \
    ln -s /opt/hailo/usr/lib/libhailort.so.${HAILOVERSION} /usr/lib/libhailort.so
ENV LD_LIBRARY_PATH="/opt/hailo/usr/lib"
ENV PATH="$PATH:/opt/hailo/usr/bin"

# Setup Python Environment
WORKDIR /tmp
ARG REQ_FILE="requirements_${HWARCH}.txt"
COPY $REQ_FILE /tmp
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
RUN ln -sf /opt/venv/bin/python3 /opt/venv/bin/python
RUN pip install -Ur /tmp/$REQ_FILE