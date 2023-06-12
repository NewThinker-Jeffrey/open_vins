

set(HPS_3RDPARTY_DIR /root/workspace/cc_ws/hps_ws/src/hps_3rdparty)
#set(HPS_3RDPARTY_DIR $ENV{HPS_3RDPARTY_DIR})
message(STATUS "HPS_3RDPARTY_DIR: ${HPS_3RDPARTY_DIR}")

set(EIGEN3_INCLUDE_DIR ${HPS_3RDPARTY_DIR}/eigen3.4.0/include)
set(EIGEN3_VERSION_STRING 3.4.0)
set(OpenCV_INCLUDE_DIR ${HPS_3RDPARTY_DIR}/opencv4.2.0/include)
set(OpenCV_DEP_LIBRARIES gtk-x11-2.0 gdk-x11-2.0 z cairo gdk_pixbuf-2.0 jpeg png16 webp gflags glog tiff)
set(Suitesparse_LIBRARIES cholmod spqr cxsparse amd colamd suitesparseconfig ccolamd camd metis lapack f77blas atlas)
set(Ceres_DEP_LIBRARIES ${Suitesparse_LIBRARIES})
set(OpenCV_LIBRARIES
    ${OpenCV_DEP_LIBRARIES}
    opencv_aruco
    opencv_hdf
    opencv_sfm
    opencv_bgsegm
    opencv_hfs
    opencv_shape
    opencv_bioinspired
    # opencv_highgui
    opencv_stereo
    opencv_calib3d
    opencv_img_hash
    opencv_stitching
    opencv_ccalib
    opencv_imgcodecs
    opencv_structured_light
    opencv_core
    opencv_imgproc
    opencv_superres
    opencv_datasets
    opencv_line_descriptor
    opencv_surface_matching
    opencv_dnn
    opencv_ml
    opencv_text
    opencv_dnn_objdetect
    opencv_objdetect
    opencv_tracking
    opencv_dnn_superres
    opencv_optflow
    opencv_video
    opencv_dpm
    opencv_phase_unwrapping
    # opencv_videoio
    opencv_face
    opencv_photo
    opencv_videostab
    opencv_features2d
    opencv_plot
    opencv_viz
    opencv_flann
    opencv_quality
    opencv_xfeatures2d
    opencv_freetype
    opencv_reg
    opencv_ximgproc
    opencv_fuzzy
    opencv_rgbd
    opencv_xobjdetect
    opencv_gapi
    opencv_saliency
    opencv_xphoto)

set(OpenCV_FOUND TRUE)
set(CERES_INCLUDE_DIRS ${HPS_3RDPARTY_DIR}/ceres2.0.0/include)
set(CERES_LIBRARIES ${Ceres_DEP_LIBRARIES} ceres)
#set(Ceres_FOUND TRUE)

# set(Boost_INCLUDE_DIRS ${HPS_3RDPARTY_DIR}/slam_dep_libs/boost_1_81_0/include)
# set(Boost_LIBRARIES boost_system boost_filesystem boost_thread boost_date_time)


link_directories(
    ${HPS_3RDPARTY_DIR}/opencv4.2.0/lib_arm
    ${HPS_3RDPARTY_DIR}/ceres2.0.0/lib_arm
    ${HPS_3RDPARTY_DIR}/png/lib_arm
    ${HPS_3RDPARTY_DIR}/jpeg/lib_arm
    ${HPS_3RDPARTY_DIR}/webp/lib_arm
    ${HPS_3RDPARTY_DIR}/tiff/lib_arm
    ${HPS_3RDPARTY_DIR}/gflags2.2.2/lib_arm    
    ${HPS_3RDPARTY_DIR}/glog0.6.0/lib_arm
    # ${HPS_3RDPARTY_DIR}/slam_dep_libs/boost_1_81_0/lib_arm    
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/libgtk2.0-0/lib_arm
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/libz1/lib_arm
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/libcairo2/lib_arm
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/libgdk-pixbuf-2.0/lib_arm
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/liblapack3/lib_arm
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/libatlas3-base/lib_arm    
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/suitesparse/lib_arm    
    ${HPS_3RDPARTY_DIR}/slam_dep_libs/metis5/lib_arm    
    /root/workspace/cc_ws/sysroot_docker/usr/lib)
include_directories(
    ${OpenCV_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${Ceres_INCLUDE_DIR}
    # ${Boost_INCLUDE_DIRS}
    /root/workspace/cc_ws/sysroot_docker/usr/include)