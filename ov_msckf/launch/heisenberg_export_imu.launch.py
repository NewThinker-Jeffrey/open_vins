from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import os
import sys

launch_args = [
    DeclareLaunchArgument(name="namespace", default_value="ov_msckf", description="namespace"),
    DeclareLaunchArgument(
        name="ov_enable", default_value="true", description="enable OpenVINS node"
    ),
    DeclareLaunchArgument(
        name="bagplay_enable", default_value="true", description="enable bag play node"
    ),
    DeclareLaunchArgument(
        name="bagplay_rate", default_value="1.0", description="bagplay_rate"
    ),
    DeclareLaunchArgument(
        name="dataset",
        default_value="/home/isaac/Work/datasets/Heisenberg/visual_inertial_0.5m_s_0328",
        description="path to your dataset (heisenberg datasets)",
    ),
    DeclareLaunchArgument(
        name="verbosity",
        default_value="INFO",
        description="ALL, DEBUG, INFO, WARNING, ERROR, SILENT",
    ),
]

def launch_setup(context):
    dataset = LaunchConfiguration("dataset").perform(context)
    bagplay_rate = LaunchConfiguration("bagplay_rate").perform(context)
    print("dataset: {}".format(dataset))
    node1 = Node(
        package="ov_msckf",
        executable="heisenberg_export_imu",
        condition=IfCondition(LaunchConfiguration("ov_enable")),
        namespace=LaunchConfiguration("namespace"),
        output='screen',
        parameters=[
            {"verbosity": LaunchConfiguration("verbosity")},
            {"heisenberg_dataset": dataset},
        ],
    )

    node2 = ExecuteProcess(
        cmd=[[
            'sleep 5 && ros2 bag play {}'.format(os.path.join(dataset, "vio_gt.bag"))
        ]],
        shell=True,
        condition=IfCondition(LaunchConfiguration("bagplay_enable")),
    )

    return [node2, node1]


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld
