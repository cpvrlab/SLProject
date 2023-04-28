#
# CMake project definition for exercise cv13_FacialLandmarkDetection
#

sl_add_exercise(cv13_FacialLandmarkDetection
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv13_FacialLandmarkDetection.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
