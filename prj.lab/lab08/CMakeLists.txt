cmake_minimum_required(VERSION 3.10)
project(task08)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(task08 task08.cpp)
target_link_libraries(task08 ${OpenCV_LIBS})