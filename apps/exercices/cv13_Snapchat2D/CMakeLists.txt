#
# CMake project definition for exercise cv13_Snapchat2D
#

sl_add_exercise(cv13_Snapchat2D
    HEADERS
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cv13_Snapchat2D.cpp
    SHADERS
    LIBRARIES
    sl_cv
    ${glfw_LIBS}
    )
