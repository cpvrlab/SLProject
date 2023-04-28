#
# CMake project definition for exercise cv06_WarpTriangle
#

sl_add_exercise(cv06_WarpTriangle
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv06_WarpTriangle.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
