cmake_minimum_required(VERSION 3.10)
project(lab04)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(task04_01 task04_01.cpp)
target_link_libraries(task04_01 PRIVATE 
    opencv_core 
    opencv_imgproc 
    opencv_highgui
    nlohmann_json::nlohmann_json
)

add_executable(task04_02 task04_02.cpp)
target_link_libraries(task04_02 PRIVATE 
    opencv_core 
    opencv_imgproc 
    opencv_highgui
    nlohmann_json::nlohmann_json
)

add_executable(task04_03 task04_03.cpp)
target_link_libraries(task04_03 PRIVATE 
    opencv_core 
    opencv_imgproc 
    opencv_highgui
    nlohmann_json::nlohmann_json
)