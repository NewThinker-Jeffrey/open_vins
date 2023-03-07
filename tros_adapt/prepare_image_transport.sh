# ./build.sh <package_name>

my_dir=$(cd $(dirname $0) && pwd)
#echo $my_dir

cd $my_dir/../..  # cd ${WorkSpace}/src
git clone --recursive https://github.com/ros-perception/image_common.git
cd image_common
git reset --hard
git checkout foxy

