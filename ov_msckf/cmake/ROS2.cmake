cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(ament_cmake REQUIRED)

find_package(ov_core REQUIRED)
find_package(ov_init REQUIRED)
#find_package(glog_catkin REQUIRED)

# find_package(realsense2)
find_package(realsense2 REQUIRED)


# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
    
#     find_package(gflags REQUIRED)
#     message("gflags_LIBRARIES: ${gflags_LIBRARIES}")
#     include_directories(
#             ${gflags_INCLUDE_DIR}
#     )
#     list(APPEND thirdparty_libraries
#            ${gflags_LIBRARIES}
#     )
else()
    add_definitions(-DROS_AVAILABLE=2)

    # Find ros dependencies
    find_package(rclcpp REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(tf2_geometry_msgs REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(image_transport REQUIRED)

    list(APPEND ament_libraries
                rclcpp
                tf2_ros
                tf2_geometry_msgs
                std_msgs
                geometry_msgs
                sensor_msgs
                nav_msgs
                cv_bridge
                image_transport
    )
endif ()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        # ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        # ${Boost_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Pangolin_LIBRARIES}
)

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/core/VioManager.cpp
        src/core/VioManagerHelper.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp
        src/ov_interface/VIO.cpp
        src/slam_dataset/vi_capture.cpp
        src/slam_dataset/vi_player.cpp
        src/slam_dataset/vi_recorder.cpp        
        src/slam_viz/pangolin_helper.cpp
        src/no_ros/VisualizerHelper.cpp
        src/no_ros/Viewer.cpp
        src/no_ros/VisualizerForViCapture.cpp
        )


if(realsense2_FOUND)    
    include_directories(${realsense_INCLUDE_DIR})
    list(APPEND LIBRARY_SOURCES
        src/slam_dataset/rs/rs_capture.cpp
        src/slam_dataset/rs/rs_helper.cpp
    )
    list(APPEND thirdparty_libraries
        ${realsense2_LIBRARY}
    )
endif()

file(GLOB_RECURSE LIBRARY_HEADERS "src/core/*.h" "src/ov_interface/*.h" "src/state/*.h" "src/update/*.h" "src/utils/*.h" "src/sim/*.h" "src/no_ros/*.h")
add_library(ov_msckf_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
# ament_target_dependencies(ov_msckf_lib ${ament_libraries} ov_core ov_init)
ament_target_dependencies(ov_msckf_lib ov_core ov_init)
target_link_libraries(ov_msckf_lib ${thirdparty_libraries})
target_include_directories(ov_msckf_lib PUBLIC src/)
install(TARGETS ov_msckf_lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include)
ament_export_libraries(ov_msckf_lib)

##################################################
# Make binary files!
##################################################

list(APPEND vicapture_msckf_SOURCES
                src/run_vicapture_msckf.cpp
                )

if (ENABLE_ROS)
        # ROS_AVAILABLE=2
        list(APPEND vicapture_msckf_SOURCES
                src/ros/ROS2Visualizer.cpp
                src/ros/ROS2Visualizer.h
                src/ros/ROSVisualizerHelper.cpp
                src/ros/ROSVisualizerHelper.h
                src/ros/ROS2VisualizerForViCapture.cpp
                src/ros/ROS2VisualizerForViCapture.h
                )
endif()

add_executable(run_vicapture_msckf 
               ${vicapture_msckf_SOURCES}
               )
               
ament_target_dependencies(run_vicapture_msckf ${ament_libraries} ov_core ov_init)
target_link_libraries(run_vicapture_msckf ov_msckf_lib ${thirdparty_libraries})
install(TARGETS run_vicapture_msckf DESTINATION lib/${PROJECT_NAME})

if (ENABLE_ROS)
        # ROS_AVAILABLE=2

        add_executable(run_subscribe_msckf
                src/run_subscribe_msckf.cpp
                src/ros/ROS2Visualizer.cpp
                src/ros/ROS2Visualizer.h
                src/ros/ROSVisualizerHelper.cpp
                src/ros/ROSVisualizerHelper.h
                )
        ament_target_dependencies(run_subscribe_msckf ${ament_libraries} ov_core ov_init)
        target_link_libraries(run_subscribe_msckf ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS run_subscribe_msckf DESTINATION lib/${PROJECT_NAME})
        
        add_executable(run_simulation 
                        src/run_simulation.cpp
                        src/ros/ROS2Visualizer.cpp
                        src/ros/ROS2Visualizer.h
                        src/ros/ROSVisualizerHelper.cpp
                        src/ros/ROSVisualizerHelper.h
                        )
        ament_target_dependencies(run_simulation ${ament_libraries} ov_core ov_init)
        target_link_libraries(run_simulation ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS run_simulation DESTINATION lib/${PROJECT_NAME})
        
        add_executable(test_sim_meas src/test_sim_meas.cpp)
        ament_target_dependencies(test_sim_meas ${ament_libraries} ov_core ov_init)
        target_link_libraries(test_sim_meas ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS test_sim_meas DESTINATION lib/${PROJECT_NAME})
        
        add_executable(test_sim_repeat src/test_sim_repeat.cpp)
        ament_target_dependencies(test_sim_repeat ${ament_libraries} ov_core ov_init)
        target_link_libraries(test_sim_repeat ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS test_sim_repeat DESTINATION lib/${PROJECT_NAME})
        
endif()



# Install launch and config directories
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)
install(DIRECTORY ../config/ DESTINATION share/${PROJECT_NAME}/config/)

# finally define this as the package
ament_package()
