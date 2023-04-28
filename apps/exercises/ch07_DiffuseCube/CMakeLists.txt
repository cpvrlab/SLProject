#
# CMake project definition for exercise ch07_DiffuseCube
#

sl_add_exercise(ch07_DiffuseCube
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DiffuseCube.cpp
    SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/Diffuse.frag
    ${SL_PROJECT_ROOT}/data/shaders/Diffuse.vert
    LIBRARIES
    sl_utils
    sl_external
    sl_cv
    ${glfw_LIBS}
    )
