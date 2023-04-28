#
# CMake project definition for exercise ch09_TextureMapping
#

sl_add_exercise(ch09_TextureMapping
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
    SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/ADSTex.frag
    ${SL_PROJECT_ROOT}/data/shaders/ADSTex.vert
    LIBRARIES
    sl_utils
    sl_external
    sl_cv
    sl_math
    ${glfw_LIBS}
    )
