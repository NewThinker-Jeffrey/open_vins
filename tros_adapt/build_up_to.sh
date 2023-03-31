# ./build.sh <package_name>

my_dir=$(cd $(dirname $0) && pwd)
cd $my_dir/../../..  # cd ${WorkSpace}/

source /opt/tros/setup.bash

export TARGET_ARCH=aarch64
export TARGET_TRIPLE=aarch64-linux-gnu
export CROSS_COMPILE=/usr/bin/$TARGET_TRIPLE-

colcon build \
    --parallel-workers 4 \
    --merge-install \
    --symlink-install \
    --cmake-force-configure \
    --cmake-args \
    --no-warn-unused-cli \
    -DOPENVINS_FOR_TROS=ON \
    -DCMAKE_TOOLCHAIN_FILE="/root/workspace/cc_ws/hps_ws/robot_dev_config/aarch64_toolchainfile.cmake" --packages-up-to $@
