#
# CMake project definition for exercise cv02_CalderonFilter
#

sl_add_exercise(cv02_CalderonFilter
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv02_CalderonFilter.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
