add_executable(task03 task03.cpp)
target_link_libraries(task03 semcv ${OpenCV_LIBS} ${TIFF_LIBRARIES})