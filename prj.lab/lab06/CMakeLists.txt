cmake_minimum_required(VERSION 3.10)
project(task06)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(task06
    task06.cpp
)

target_include_directories(task06 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(task06 PRIVATE
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_features2d
    semcv
    nlohmann_json::nlohmann_json
)

set_target_properties(task06 PROPERTIES OUTPUT_NAME "task06")