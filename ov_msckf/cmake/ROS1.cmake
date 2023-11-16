cmake_minimum_required(VERSION 3.3)

option(USE_HEAR_SLAM "Enable or disable building with hear_slam" OFF)
if (USE_HEAR_SLAM)
    set(HEAR_SLAM_PKG hear_slam)
    add_definitions(-DUSE_HEAR_SLAM)
    message(STATUS "Will use hear_slam!!")
else()
    message(STATUS "Won't use hear_slam.")
endif()

# Find ROS build system
find_package(catkin QUIET COMPONENTS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge ov_core ov_init ${HEAR_SLAM_PKG})

# find_package(realsense2)
find_package(realsense2 REQUIRED)


# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge ov_core ov_init ${HEAR_SLAM_PKG}
            INCLUDE_DIRS src/
            LIBRARIES ov_msckf_lib
    )
else ()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
    include(GNUInstallDirs)
    set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif ()


# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        # ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        # ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${catkin_LIBRARIES}
)

# If we are not building with ROS then we need to manually link to its headers
# This isn't that elegant of a way, but this at least allows for building without ROS
# If we had a root cmake we could do this: https://stackoverflow.com/a/11217008/7718197
# But since we don't we need to basically build all the cpp / h files explicitly :(
if (NOT catkin_FOUND OR NOT ENABLE_ROS)

    message(STATUS "MANUALLY LINKING TO OV_CORE LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_core/src/)
    file(GLOB_RECURSE OVCORE_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/../ov_core/src/*.cpp")
    list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_webcam\\.cpp$")
    list(FILTER OVCORE_LIBRARY_SOURCES EXCLUDE REGEX ".*test_tracking\\.cpp$")
    list(APPEND LIBRARY_SOURCES ${OVCORE_LIBRARY_SOURCES})
    file(GLOB_RECURSE OVCORE_LIBRARY_HEADERS "${CMAKE_SOURCE_DIR}/../ov_core/src/*.h")
    list(APPEND LIBRARY_HEADERS ${OVCORE_LIBRARY_HEADERS})

    message(STATUS "MANUALLY LINKING TO OV_INIT LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_init/src/)
    file(GLOB_RECURSE OVINIT_LIBRARY_SOURCES "${CMAKE_SOURCE_DIR}/../ov_init/src/*.cpp")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_init\\.cpp$")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_dynamic_mle\\.cpp$")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*test_simulation\\.cpp$")
    list(FILTER OVINIT_LIBRARY_SOURCES EXCLUDE REGEX ".*Simulator\\.cpp$")
    list(APPEND LIBRARY_SOURCES ${OVINIT_LIBRARY_SOURCES})
    file(GLOB_RECURSE OVINIT_LIBRARY_HEADERS "${CMAKE_SOURCE_DIR}/../ov_init/src/*.h")
    list(FILTER OVINIT_LIBRARY_HEADERS EXCLUDE REGEX ".*Simulator\\.h$")
    list(APPEND LIBRARY_HEADERS ${OVINIT_LIBRARY_HEADERS})

endif ()

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
    
message(STATUS "LIBRARY_SOURCES AND LIBRARY_HEADERS:  ${LIBRARY_SOURCES} ${LIBRARY_HEADERS}")

# if (catkin_FOUND AND ENABLE_ROS)
#     list(APPEND LIBRARY_SOURCES src/ros/ROS1Visualizer.cpp src/ros/ROSVisualizerHelper.cpp)
# endif ()
# file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
file(GLOB_RECURSE LIBRARY_HEADERS "src/core/*.h" "src/ov_interface/*.h" "src/state/*.h" "src/update/*.h" "src/utils/*.h" "src/sim/*.h" "src/no_ros/*.h")
add_library(ov_msckf_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(ov_msckf_lib ${thirdparty_libraries})
target_include_directories(ov_msckf_lib PUBLIC src/)
install(TARGETS ov_msckf_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##################################################
# Make binary files!
##################################################

list(APPEND vicapture_msckf_SOURCES
        src/run_vicapture_msckf.cpp
        src/ros/ROS1Visualizer.cpp
        src/ros/ROS1Visualizer.h
        src/ros/ROSVisualizerHelper.cpp
        src/ros/ROSVisualizerHelper.h
        src/ros/ROS1VisualizerForViCapture.cpp
        src/ros/ROS1VisualizerForViCapture.h                
        )

if (ENABLE_ROS)
        # ROS_AVAILABLE=1
        list(APPEND vicapture_msckf_SOURCES
                )
endif()



add_executable(run_vicapture_msckf
                ${vicapture_msckf_SOURCES})

target_link_libraries(run_vicapture_msckf ov_msckf_lib ${thirdparty_libraries})
install(TARGETS run_vicapture_msckf
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
            
if (catkin_FOUND AND ENABLE_ROS)

    add_executable(ros1_serial_msckf
                src/ros1_serial_msckf.cpp
                src/ros/ROS1Visualizer.cpp
                src/ros/ROS1Visualizer.h
                src/ros/ROSVisualizerHelper.cpp
                src/ros/ROSVisualizerHelper.h
                )

    target_link_libraries(ros1_serial_msckf ov_msckf_lib ${thirdparty_libraries})
    install(TARGETS ros1_serial_msckf
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

    add_executable(run_subscribe_msckf
                   src/run_subscribe_msckf.cpp
                   src/ros/ROS1Visualizer.cpp
                   src/ros/ROS1Visualizer.h
                   src/ros/ROSVisualizerHelper.cpp
                   src/ros/ROSVisualizerHelper.h
                   )
    target_link_libraries(run_subscribe_msckf ov_msckf_lib ${thirdparty_libraries})
    install(TARGETS run_subscribe_msckf
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

        add_executable(run_simulation
                src/run_simulation.cpp
                src/ros/ROS1Visualizer.cpp
                src/ros/ROS1Visualizer.h
                src/ros/ROSVisualizerHelper.cpp
                src/ros/ROSVisualizerHelper.h
                )
        target_link_libraries(run_simulation ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS run_simulation
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
                )

        add_executable(test_sim_meas src/test_sim_meas.cpp)
        target_link_libraries(test_sim_meas ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS test_sim_meas
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
                )

        add_executable(test_sim_repeat src/test_sim_repeat.cpp)
        target_link_libraries(test_sim_repeat ov_msckf_lib ${thirdparty_libraries})
        install(TARGETS test_sim_repeat
                ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
                RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
                )
endif ()



##################################################
# Launch files!
##################################################

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)





