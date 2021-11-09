cmake_minimum_required(VERSION 3.10)
project(OptimizedFishRecognizer)

include_directories(src/include)
file(GLOB SOURCES "src/*.cpp")

add_executable(fishspecies fishspecies.cpp ${SOURCES})

target_link_libraries(fishspecies nvinfer nvinfer_plugin nvonnxparser "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so.10.2")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
target_link_libraries(fishspecies opencv_core opencv_calib3d opencv_shape opencv_stitching opencv_superres opencv_videostab opencv_aruco opencv_bgsegm opencv_bioinspired opencv_ccalib opencv_datasets opencv_dpm opencv_face opencv_freetype opencv_fuzzy opencv_line_descriptor opencv_optflow opencv_video opencv_plot opencv_reg opencv_saliency opencv_stereo opencv_structured_light opencv_phase_unwrapping opencv_rgbd  opencv_surface_matching  opencv_ximgproc opencv_calib3d opencv_features2d opencv_flann opencv_xobjdetect opencv_objdetect opencv_hdf opencv_viz opencv_text opencv_ml opencv_xphoto opencv_highgui opencv_videoio opencv_imgcodecs opencv_photo opencv_imgproc)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
