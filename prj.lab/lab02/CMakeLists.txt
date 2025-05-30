project(lab02)

find_package(OpenCV REQUIRED)
add_executable(task02 task02.cpp)
target_link_libraries(task02 semcv ${OpenCV_LIBS})