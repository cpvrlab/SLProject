#
# CMake project definition for exercise cv13_FaceTracking
#

sl_add_exercise(cv13_FaceTracking
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv13_FaceTracking.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
