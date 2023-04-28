#
# CMake project definition for exercise cv13_HeadPoseEstimation
#

sl_add_exercise(cv13_HeadPoseEstimation
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv13_HeadPoseEstimation.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
