cmake_minimum_required(VERSION 2.8)

project(facial_landmark_point_detction)
set (CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "/home/jps/opencv_wit_tes/opencv-3.4.0/build/install/share/OpenCV")
set(dlib_DIR "/home/jps/LIBRARIES/dlib/dlib/new_new_build/new_new_install/lib/cmake/dlib")

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn features2d tracking calib3d xfeatures2d objdetect aruco)
find_package(dlib REQUIRED COMPONENTS dlib)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${dlib_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}  ${dlib_LIBS})
