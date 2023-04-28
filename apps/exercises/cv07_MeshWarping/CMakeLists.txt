#
# CMake project definition for exercise cv07_MeshWarping
#

sl_add_exercise(cv07_MeshWarping
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv07_MeshWarping.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
