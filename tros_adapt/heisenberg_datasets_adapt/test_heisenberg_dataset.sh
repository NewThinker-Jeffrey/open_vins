#!/usr/bin/env bash

my_dir=$(cd $(dirname $0) && pwd)
# echo $my_dir
source  $my_dir/../../../../install/setup.bash


if [ "$1" != "" ]; then
    dataset="$1"
else
    dataset=""
fi

if [ "$dataset" != "" ]; then
    ros2 launch ov_msckf heisenberg_subscribe.launch.py dataset:=$dataset
else
    ros2 launch ov_msckf heisenberg_subscribe.launch.py
fi
