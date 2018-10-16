#
# CMake project definition for exercise ch09_TextureMapping
#

set(target ch09_TextureMapping)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/glUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/SLCV.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/SLCVImage.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/glUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SL.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/SL/SLFileSystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib-SLProject/source/CV/SLCVImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
    )

file(GLOB shaders
    ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/shaders/ADSTex.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/../../_data/shaders/ADSTex.vert
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/dirent
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/glfw3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/lib-SLExternal/DBoW2
    ${OpenCV_INCLUDE_DIR}
    ${g2o_INCLUDE_DIR}
    PUBLIC
    INTERFACE
    )

include(../../cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE
    ${PlatformLinkLibs}
    ${META_PROJECT_NAME}::lib-SLExternal
    ${OpenCV_LIBS}
    ${g2o_LIBS}
    PUBLIC
    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
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
