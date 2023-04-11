# ./build.sh <package_name>

my_dir=$(cd $(dirname $0) && pwd)
#echo $my_dir

cd $my_dir/../..  # cd ${WorkSpace}/src
git clone --recursive https://github.com/ros-perception/image_transport_plugins.git
cd image_transport_plugins
git reset --hard
git checkout foxy-devel

