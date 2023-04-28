#
# CMake project definition for ParticleGenerator
#

sl_add_exercise(ParticleGenerator
    HEADERS
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.h
    SOURCES
    ${SL_PROJECT_ROOT}/apps/exercises/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ParticleGenerator.cpp
    SHADERS
    ${SL_PROJECT_ROOT}/data/shaders/ParticleOLD.vert
    ${SL_PROJECT_ROOT}/data/shaders/ParticleOLD.frag
    ${SL_PROJECT_ROOT}/data/shaders/ParticleOLD.geom
    ${SL_PROJECT_ROOT}/data/shaders/ParticleTFOLD.vert
    ${SL_PROJECT_ROOT}/data/shaders/ParticleTFOLD.frag
    ${SL_PROJECT_ROOT}/data/shaders/ColorAttribute.vert
    ${SL_PROJECT_ROOT}/data/shaders/Color.frag
    LIBRARIES
    sl_external
    sl_utils
    sl_cv
    ${glfw_LIBS}
    )
