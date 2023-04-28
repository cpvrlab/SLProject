#
# CMake project definition for exercise cv08_MeshMorphing
#

sl_add_exercise(cv08_MeshMorphing
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv08_MeshMorphing.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
