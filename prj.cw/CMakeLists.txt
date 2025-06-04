find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(main_cw main_cw.cpp)
add_executable(detect_angle_new testing/detect_angle_new.cpp)

target_link_libraries(main_cw ${OpenCV_LIBS})
target_link_libraries(detect_angle_new ${OpenCV_LIBS} nlohmann_json::nlohmann_json)