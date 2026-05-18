#!/bin/bash

ENV_CONFIG_FILE="/etc/environment"
ENV_PROXY_CONFIG='
http_proxy=http://cloudproxy.zf-world.com:8080
https_proxy=http://cloudproxy.zf-world.com:8080
no_proxy=127.0.0.1,localhost,login.cloud.zf.com,*.zf-world.com,10.0.0.0/8
'

DOCKER_CONFIG_ROOT="/etc/systemd/system/docker.service.d/"
DOCKER_CONFIG_FILE="$DOCKER_CONFIG_ROOT/http-proxy.conf"
DOCKER_PROXY_CONFIG='
[Service]
Environment="http_proxy=http://cloudproxy.zf-world.com:8080"
Environment="https_proxy=http://cloudproxy.zf-world.com:8080"
Environment="no_proxy=zf-world.com,trw.com,login.cloud.zf.com,corp.transics.com,westeurope.aroapp.io,localhost"
'

function configureProxy()
{
    sudo mkdir -p $DOCKER_CONFIG_ROOT
    sudo touch $DOCKER_CONFIG_FILE
    
    for LINE in $ENV_PROXY_CONFIG; do
        if ! grep -q $LINE $ENV_CONFIG_FILE; then
            echo $LINE | sudo tee -a $ENV_CONFIG_FILE
        fi
    done
    
    for LINE in "$DOCKER_PROXY_CONFIG"; do
        if ! grep -q "$LINE" $DOCKER_CONFIG_FILE; then
            echo "$LINE" | sudo tee -a $DOCKER_CONFIG_FILE
        fi
    done

    source $ENV_CONFIG_FILE
}

function configureSystem()
{
    SYS_DEPS="linux-headers-$(uname -r) linux-image-$(uname -r)"
    TOOL_DEPS="git wget curl software-properties-common ca-certificates"
    BUILD_DEPS="gcc build-essential cmake ninja-build pkg-config"
    XSERVER_DEPS="xorg openbox xauth"
    sudo apt update -y && sudo apt-get upgrade -y
    sudo apt install -y $SYS_DEPS $TOOL_DEPS $BUILD_DEPS

    xhost +
}

function configureLocale()
{
    echo 'keyboard-configuration  keyboard-configuration/layoutcode  select  de' | sudo debconf-set-selections
    echo 'keyboard-configuration  keyboard-configuration/xkb-keymap  select  de' | sudo debconf-set-selections
    sudo sed -i 's/XKBLAYOUT=.*/XKBLAYOUT="de"/' /etc/default/keyboard
    sudo dpkg-reconfigure -f noninteractive keyboard-configuration
    sudo udevadm trigger --subsystem-match=input --action=change
}

function installDocker()
{
    sudo mkdir -p /tmp
    sudo curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo chmod +x /tmp/get-docker.sh && sh /tmp/get-docker.sh
    sudo groupadd docker && sudo usermod -aG docker $USER
    sudo systemctl daemon-reload
    sudo systemctl restart docker
}

function installDriver()
{
    sudo dpkg -i hailort_4.23.0_arm64.deb
    sudo dpkg -i hailort-pcie-driver_4.23.0_all.deb
}

################################################################################################
echo "[CONFIGURING PROXY] Start."
configureProxy
echo "[CONFIGURING PROXY] Done."
echo "[CONFIGURING SYSTEM] Start."
configureSystem
echo "[CONFIGURING SYSTEM] Done."
echo "[CONFIGURING LOCALES] Start."
configureLocale
echo "[CONFIGURING LOCALES] Done."
echo "[CONFIGURING DOCKER] Start."
installDocker
echo "[CONFIGURING DOCKER] Done."
echo "[CONFIGURING HDRIVER] Start."
installDriver
echo "[CONFIGURING HDRIVER] Done."