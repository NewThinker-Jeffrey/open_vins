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
        name="rviz_enable", default_value="false", description="enable rviz node"
    ),
    DeclareLaunchArgument(
        name="play_rate", default_value="1.0", description="play_rate"
    ),
    DeclareLaunchArgument(
        name="dataset",
        default_value="",
        description="path to your dataset",
    ),
    DeclareLaunchArgument(
        name="config",
        default_value="rs_d435i",
        description="rs_d435i, ...",
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
    # DeclareLaunchArgument(
    #     name="use_stereo",
    #     default_value="false",
    #     description="if we have more than 1 camera, if we should try to track stereo constraints between pairs",
    # ),
    # DeclareLaunchArgument(
    #     name="klt_left_major_stereo",
    #     default_value="true",
    #     description="whether to enable left_major_stereo. only valid for klt tracking.",
    # ),
    # DeclareLaunchArgument(
    #     name="klt_strict_stereo",
    #     default_value="false",
    #     description="whether to enable strict_stereo. only valid when klt_left_major_stereo=true.",
    # ),
    # DeclareLaunchArgument(
    #     name="klt_force_fundamental",
    #     default_value="true",
    #     description="If true, use fundamental check instead of essential check",
    # ),
    # DeclareLaunchArgument(
    #     name="feattrack_high_frequency_log",
    #     default_value="false",
    #     description="If true, enable high_frequency debug log for feature tracking",
    # ),    
    # DeclareLaunchArgument(
    #     name="feattrack_predict_keypoints",
    #     default_value="true",
    #     description="If true, enable keypoint prediction (based on imu) for feature tracking",
    # ),    
    # DeclareLaunchArgument(
    #     name="vio_manager_high_frequency_log",
    #     default_value="false",
    #     description="If true, enable high_frequency debug log for feature tracking",
    # ),    
    # DeclareLaunchArgument(
    #     name="choose_new_landmark_by_disparity",
    #     default_value="false",
    #     description="sort max_track features by their disparities and choose ones with larger disparities as new landmarks",
    # ),    
    # DeclareLaunchArgument(
    #     name="enable_early_landmark",
    #     default_value="false",
    #     description="enable_early_landmark (features those are not of max_track can also be chosen as landmarks if there're too few landmarks)",
    # ),    
    # DeclareLaunchArgument(
    #     name="max_cameras",
    #     default_value="1",
    #     description="how many cameras we have 1 = mono, 2 = stereo, >2 = binocular (all mono tracking)",
    # ),
    DeclareLaunchArgument(
        name="output_dir",
        default_value="",
        description="path to the output",
    ),
    DeclareLaunchArgument(
        name="save_feature_images",
        default_value="false",
        description="record the feature images",
    ),
    DeclareLaunchArgument(
        name="record_timing_information",
        default_value="false",
        description="record the record_timing_information",
    ),
    DeclareLaunchArgument(
        name="save_total_state",
        default_value="false",
        description="record the total state with calibration and features to a txt file",
    )
]

def launch_setup(context):
    config_path = LaunchConfiguration("config_path").perform(context)
    output_dir = LaunchConfiguration("output_dir").perform(context)
    dataset = LaunchConfiguration("dataset").perform(context)
    play_rate = LaunchConfiguration("play_rate").perform(context)
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
        executable="run_vicapture_msckf",
        condition=IfCondition(LaunchConfiguration("ov_enable")),
        namespace=LaunchConfiguration("namespace"),
        output='screen',
        parameters=[
            {"verbosity": LaunchConfiguration("verbosity")},
            # {"use_stereo": LaunchConfiguration("use_stereo")},
            # {"klt_left_major_stereo": LaunchConfiguration("klt_left_major_stereo")},
            # {"klt_strict_stereo": LaunchConfiguration("klt_strict_stereo")},
            # {"klt_force_fundamental": LaunchConfiguration("klt_force_fundamental")},
            # {"feattrack_high_frequency_log": LaunchConfiguration("feattrack_high_frequency_log")},
            # {"feattrack_predict_keypoints": LaunchConfiguration("feattrack_predict_keypoints")},
            # {"vio_manager_high_frequency_log": LaunchConfiguration("vio_manager_high_frequency_log")},       
            # {"choose_new_landmark_by_disparity": LaunchConfiguration("choose_new_landmark_by_disparity")},       
            # {"enable_early_landmark": LaunchConfiguration("enable_early_landmark")},       
            # {"max_cameras": LaunchConfiguration("max_cameras")},
            {"record_timing_information": LaunchConfiguration("record_timing_information")},
            {"save_total_state": LaunchConfiguration("save_total_state")},
            {"save_feature_images": LaunchConfiguration("save_feature_images")},
            {"config_path": config_path},
            {"dataset": dataset},
            {"output_dir": output_dir},
            {"play_rate": play_rate},
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

    return [node1, node2]


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld
