#!/bin/bash

set -ex
PROJECT_HOME=/media/disk1/framework-sim/yang/
useradd yang
passwd yang
yum install -y sudo python3-pip
pip install virtualenv
sed -i "42s/^/#/g" /etc/ssh/ssh_config
source $PROJECT_HOME/set_proxy.sh set
bash $PROJECT_HOME/tools/install-necessarities.sh

mkdir -p /data/yang/software
mkdir -p /data/common

export CMAKE_PATH=${PROJECT_HOME}/tools/cmake-3.22.1-linux-x86_64/
export PATH=${CMAKE_PATH}/bin:$PATH
