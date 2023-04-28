#
# CMake project definition for exercise cv15_YoloV3
#

sl_add_exercise(cv15_YoloV3
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv15_YoloV3.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
