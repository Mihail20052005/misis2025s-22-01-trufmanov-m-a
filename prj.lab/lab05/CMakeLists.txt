project(lab05)

find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)

add_executable(task05 task05.cpp)

target_link_libraries(task05 ${OpenCV_LIBS} ${TIFF_LIBRARIES})