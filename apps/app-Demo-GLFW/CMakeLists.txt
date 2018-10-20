# 
# CMake configuration for app-Demo-GLFW application
#

set(target app-Demo-GLFW)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions
    SL_STARTSCENE=SID_Minimal)

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoGui.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoGui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoLoad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainGLFW.cpp
    )

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "apps"
    )

target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SL_PROJECT_ROOT}/include
    ${SL_PROJECT_ROOT}/external/lib-SLExternal
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/imgui
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/spa
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/dirent
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/glew/include
    ${SL_PROJECT_ROOT}/external/lib-SLExternal/glfw3/include
    ${OpenCV_INCLUDE_DIR}
    PUBLIC
    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE
    lib-SLProject
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

target_link_libraries(${target}
    PRIVATE
    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    INTERFACE
    )

include(${SL_PROJECT_ROOT}/cmake/CopyCVLibsToTargetFolder.cmake)
