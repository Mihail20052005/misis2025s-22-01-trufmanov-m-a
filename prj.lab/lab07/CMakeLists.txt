cmake_minimum_required(VERSION 3.10)
project(lab07)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(task07-01 task07-01.cpp)
target_link_libraries(task07-01 
    PRIVATE 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

add_executable(task07-02 task07-02.cpp)
target_link_libraries(task07-02 
    PRIVATE 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)