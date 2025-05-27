cmake_minimum_required(VERSION 3.23)

add_library(semcv semcv.cpp semcv.hpp)

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
target_include_directories(semcv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(semcv PRIVATE ${OpenCV_LIBS})