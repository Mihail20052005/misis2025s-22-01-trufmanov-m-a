add_executable(main_cw main_cw.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(main_cw ${OpenCV_LIBS})
