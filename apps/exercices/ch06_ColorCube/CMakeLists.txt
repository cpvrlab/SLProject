#
# CMake project definition for exercise ch06_ColorCube
#

set(target ch06_ColorCube)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/exercices/glUtils.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/exercices/glUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ColorCube.cpp
        )

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/ColorAttribute.vert
        ${SL_PROJECT_ROOT}/data/shaders/Color.frag
        )

add_executable(${target}
        ${headers}
        ${sources}
        ${shaders}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps/exercices"
        )

target_include_directories(${target} PRIVATE ${SL_PROJECT_ROOT}/apps/exercices)

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
        PRIVATE
        ${PlatformLinkLibs}
        sl_utils
        sl_external
        sl_cv
        ${glfw_LIBS}
        PUBLIC
        INTERFACE
        )

target_compile_definitions(${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_COMPILE_DEFINITIONS}
        INTERFACE
        )

target_compile_options(${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}
        INTERFACE
        )

