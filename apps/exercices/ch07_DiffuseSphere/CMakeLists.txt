#
# CMake project definition for exercise ch07_DiffuseSphere
#

set(target ch07_DiffuseSphere)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/exercices/glUtils.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/exercices/glUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DiffuseSphere.cpp
        )

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/Diffuse.frag
        ${SL_PROJECT_ROOT}/data/shaders/Diffuse.vert
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
        ENABLE_EXPORTS ON
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

