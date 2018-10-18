#
# CMake project definition for exercise ch09_TextureMapping
#

set(target ch09_TextureMapping)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
    ${SL_PROJECT_ROOT}/include/glUtils.h
    ${SL_PROJECT_ROOT}/include/SLCV.h
    ${SL_PROJECT_ROOT}/include/SLCVImage.h
    )

file(GLOB sources
    ${SL_PROJECT_ROOT}/lib-SLProject/source/glUtils.cpp
    ${SL_PROJECT_ROOT}/lib-SLProject/source/SL/SL.cpp
    ${SL_PROJECT_ROOT}/lib-SLProject/source/SL/SLFileSystem.cpp
    ${SL_PROJECT_ROOT}/lib-SLProject/source/CV/SLCVImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
    )

file(GLOB shaders
    ${SL_PROJECT_ROOT}/_data/shaders/ADSTex.frag
    ${SL_PROJECT_ROOT}/_data/shaders/ADSTex.vert
    )

add_executable(${target}
    ${headers}
    ${sources}
    ${shaders}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    )

target_include_directories(${target}
    PRIVATE
    ${SL_PROJECT_ROOT}/include
    ${SL_PROJECT_ROOT}/external/lib-SLExternal
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/dirent
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/glew/include
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/glfw3/include
    ${OpenCV_INCLUDE_DIR}
    PUBLIC
    INTERFACE
    )

include(../../cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE
    ${PlatformLinkLibs}
    ${META_PROJECT_NAME}::lib-SLExternal
    ${OpenCV_LIBS}
    PUBLIC
    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
    GLEW_STATIC
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

include(${SL_PROJECT_ROOT}/cmake/CopyCVLibsToTargetFolder.cmake)
