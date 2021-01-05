#
# CMake project definition for exercise ch06_ColorCube
#

set(target ch06_ColorCube)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/modules/sl/include/glUtils.h
        ${SL_PROJECT_ROOT}/modules/sl/include/CVTypedefs.h
        ${SL_PROJECT_ROOT}/modules/sl/include/CVImage.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/modules/sl/source/glUtils.cpp
        ${SL_PROJECT_ROOT}/modules/sl/source/SL/SL.cpp
        ${SL_PROJECT_ROOT}/modules/sl/source/CV/CVImage.cpp
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

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/modules/math/include
        ${SL_PROJECT_ROOT}/modules/math/externals
        ${SL_PROJECT_ROOT}/modules/sl/include
        ${SL_PROJECT_ROOT}/modules/sl/include/CV
        ${SL_PROJECT_ROOT}/modules/sl/include/GL
        ${SL_PROJECT_ROOT}/modules/sl/include/SL
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

