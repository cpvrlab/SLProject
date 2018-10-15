# 
# CMake configuration for app-Demo-GLFW application
# 

set(target app-Demo-GLFW)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions SL_STARTSCENE=SID_Minimal)

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

# MSVC project options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(DEFAULT_PROJECT_OPTIONS ${DEFAULT_PROJECT_OPTIONS}
        VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
    )
endif ()
message(STATUS "DEFAULT_PROJECT_OPTIONS: ${DEFAULT_PROJECT_OPTIONS}")

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    )

target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/spa
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/dirent
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glfw3/include
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

include(../cmake/CopyCVLibsToTargetFolder.cmake)
