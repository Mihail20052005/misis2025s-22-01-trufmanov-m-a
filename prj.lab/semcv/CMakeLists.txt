add_library(semcv semcv.cpp semcv.hpp)

target_include_directories(semcv PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    $ENV{HOME}/vcpkg/installed/arm64-osx/include
)

find_package(TIFF REQUIRED)

target_link_libraries(semcv PRIVATE 
    opencv_core 
    opencv_imgproc 
    opencv_highgui
    ${TIFF_LIBRARIES}
)