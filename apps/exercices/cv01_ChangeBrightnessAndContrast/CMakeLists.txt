#
# CMake project definition for exercise cv01_ChangeBrightnessAndContrast
#

sl_add_exercise(cv01_ChangeBrightnessAndContrast
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv01_ChangeBrightnessAndContrast.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
