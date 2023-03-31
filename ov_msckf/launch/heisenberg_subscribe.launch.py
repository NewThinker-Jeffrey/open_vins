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
        name="rviz_enable", default_value="false", description="enable rviz node"
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
        name="config",
        default_value="heisenberg_1",
        description="heisenberg_1, heisenberg_2...",
    ),
    DeclareLaunchArgument(
        name="config_path",
        default_value="",
        description="path to estimator_config.yaml. If not given, determined based on provided 'config' above",
    ),
    DeclareLaunchArgument(
        name="verbosity",
        default_value="INFO",
        description="ALL, DEBUG, INFO, WARNING, ERROR, SILENT",
    ),
    DeclareLaunchArgument(
        name="use_stereo",
        default_value="false",
        description="if we have more than 1 camera, if we should try to track stereo constraints between pairs",
    ),
    DeclareLaunchArgument(
        name="max_cameras",
        default_value="1",
        description="how many cameras we have 1 = mono, 2 = stereo, >2 = binocular (all mono tracking)",
    ),
    DeclareLaunchArgument(
        name="save_total_state",
        default_value="false",
        description="record the total state with calibration and features to a txt file",
    )
]

def launch_setup(context):
    config_path = LaunchConfiguration("config_path").perform(context)
    dataset = LaunchConfiguration("dataset").perform(context)
    bagplay_rate = LaunchConfiguration("bagplay_rate").perform(context)
    print("dataset: {}".format(dataset))
    if not config_path:
        configs_dir = os.path.join(get_package_share_directory("ov_msckf"), "config")
        available_configs = os.listdir(configs_dir)
        config = LaunchConfiguration("config").perform(context)
        if config in available_configs:
            config_path = os.path.join(
                            get_package_share_directory("ov_msckf"),
                            "config",config,"estimator_config.yaml"
                        )
        else:
            return [
                LogInfo(
                    msg="ERROR: unknown config: '{}' - Available configs are: {} - not starting OpenVINS".format(
                        config, ", ".join(available_configs)
                    )
                )
            ]
    else:
        if not os.path.isfile(config_path):
            return [
                LogInfo(
                    msg="ERROR: config_path file: '{}' - does not exist. - not starting OpenVINS".format(
                        config_path)
                    )
            ]
    node1 = Node(
        package="ov_msckf",
        executable="run_subscribe_msckf",
        condition=IfCondition(LaunchConfiguration("ov_enable")),
        namespace=LaunchConfiguration("namespace"),
        output='screen',
        parameters=[
            {"verbosity": LaunchConfiguration("verbosity")},
            {"use_stereo": LaunchConfiguration("use_stereo")},
            {"max_cameras": LaunchConfiguration("max_cameras")},
            {"save_total_state": LaunchConfiguration("save_total_state")},
            {"config_path": config_path},
            {"heisenberg_dataset": dataset},
        ],
    )

    node2 = Node(
        package="rviz2",
        executable="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz_enable")),
        arguments=[
            "-d"
            + os.path.join(
                get_package_share_directory("ov_msckf"), "launch", "display_ros2.rviz"
            ),
            "--ros-args",
            "--log-level",
            "warn",
            ],
    )

    node3 = ExecuteProcess(
        cmd=[[
            'sleep 5 && ros2 bag play {}'.format(os.path.join(dataset, "vio_gt.bag"))
        ]],
        shell=True,
        condition=IfCondition(LaunchConfiguration("bagplay_enable")),
    )

    return [node3, node1, node2]


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld
