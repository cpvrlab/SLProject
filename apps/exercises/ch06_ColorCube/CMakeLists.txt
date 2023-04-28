#
# CMake project definition for exercise ch06_ColorCube
#

sl_add_exercise(ch06_ColorCube
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ColorCube.cpp
    SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/ColorAttribute.vert
    ${SL_PROJECT_ROOT}/data/shaders/Color.frag
    LIBRARIES
    sl_utils
    sl_external
    sl_cv
    ${glfw_LIBS}
    )
