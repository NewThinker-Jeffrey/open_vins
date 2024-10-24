cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(ament_cmake REQUIRED)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
#     message(FATAL_ERROR "Build with ROS1.cmake if you don't have ROS.")
else()
    add_definitions(-DROS_AVAILABLE=2)

    # Find ros dependencies
    find_package(rclcpp REQUIRED)
    find_package(cv_bridge REQUIRED)
    list(APPEND ament_libraries
                rclcpp
                cv_bridge
    )

endif ()


# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        # ${Boost_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        # ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

##################################################
# Make the core library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/cpi/CpiV1.cpp
        src/cpi/CpiV2.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackAruco.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackKLT.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
        src/utils/sensor_data.cpp
        src/utils/chi_square/chi_squared_quantile_table_0_95.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_core_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
# if (ENABLE_ROS)
#         ament_target_dependencies(ov_core_lib ${ament_libraries})
# endif()
target_link_libraries(ov_core_lib ${thirdparty_libraries})
target_include_directories(ov_core_lib PUBLIC src/)
install(TARGETS ov_core_lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include)
ament_export_libraries(ov_core_lib)

##################################################
# Make binary files!
##################################################

if (ENABLE_ROS)
        # TODO: UPGRADE THIS TO ROS2 AS ANOTHER FILE!!
        #if (catkin_FOUND AND ENABLE_ROS)
        #    add_executable(test_tracking src/test_tracking.cpp)
        #    target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
        #endif ()

        # add_executable(test_webcam src/test_webcam.cpp)
        # ament_target_dependencies(test_webcam ${ament_libraries})
        # target_link_libraries(test_webcam ov_core_lib ${thirdparty_libraries})
        # install(TARGETS test_webcam DESTINATION lib/${PROJECT_NAME})

        add_executable(test_profile src/test_profile.cpp)
        ament_target_dependencies(test_profile ${ament_libraries})
        target_link_libraries(test_profile ov_core_lib ${thirdparty_libraries})
        install(TARGETS test_profile DESTINATION lib/${PROJECT_NAME})
endif()

# finally define this as the package
ament_package()