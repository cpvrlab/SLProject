#
# CMake project definition for Geom
#

sl_add_exercise(Geom
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geom.cpp
    SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/Geom.vert
    ${SL_PROJECT_ROOT}/data/shaders/Geom.frag
    ${SL_PROJECT_ROOT}/data/shaders/Geom.geom
    LIBRARIES
    sl_external
    sl_utils
    sl_cv
    ${glfw_LIBS}
    )
