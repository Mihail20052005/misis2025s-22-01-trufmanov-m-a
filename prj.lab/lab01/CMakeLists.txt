add_executable(task01-01 task01-01.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(task01-01 semcv  ${OpenCV_LIBS})

add_executable(task01-02 task01-02.cpp)
target_link_libraries(task01-02 semcv ${OpenCV_LIBS} ${TIFF_LIBRARIES})

add_executable(generate_images generate_images.cpp)
target_link_libraries(generate_images ${OpenCV_LIBS} ${TIFF_LIBRARIES})