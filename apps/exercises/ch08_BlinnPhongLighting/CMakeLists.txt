#
# CMake project definition for exercise ch08_BlinnPhongLighting
#

sl_add_exercise(ch08_BlinnPhongLighting
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BlinnPhongLighting.cpp
    SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/ch08_BlinnPhongLighting.frag
    ${SL_PROJECT_ROOT}/data/shaders/ch08_BlinnPhongLighting.vert
    LIBRARIES
    sl_external
    sl_utils
    sl_cv
    ${glfw_LIBS}
    )
