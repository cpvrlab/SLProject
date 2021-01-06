#
# CMake project definition for exercise ch09_TextureMapping
#

set(target ch09_TextureMapping)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/lib-SLProject/include/glUtils.h
        ${SL_PROJECT_ROOT}/lib-SLProject/include/CVTypedefs.h
        ${SL_PROJECT_ROOT}/lib-SLProject/include/CVImage.h
    )

file(GLOB sources
        ${SL_PROJECT_ROOT}/lib-SLProject/source/glUtils.cpp
        ${SL_PROJECT_ROOT}/lib-SLProject/source/sl/SL.cpp
        ${SL_PROJECT_ROOT}/lib-SLProject/source/cv/CVImage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
    )

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/ADSTex.frag
        ${SL_PROJECT_ROOT}/data/shaders/ADSTex.vert
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

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/lib-SLMath/include/math
        ${SL_PROJECT_ROOT}/lib-SLProject/include
        ${SL_PROJECT_ROOT}/lib-SLProject/include/CV
        ${SL_PROJECT_ROOT}/lib-SLProject/include/GL
        ${SL_PROJECT_ROOT}/lib-SLProject/include/SL
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w/GL
        ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
        )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
        PRIVATE
        ${PlatformLinkLibs}
        lib-SLExternal
        lib-Utils
        ${OpenCV_LIBS}
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
