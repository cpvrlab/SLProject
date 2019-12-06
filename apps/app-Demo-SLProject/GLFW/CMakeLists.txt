# 
# CMake configuration for app-Demo-SLProject application
#

set(target app-Demo-SLProject)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions SL_STARTSCENE=SID_Minimal)

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoGui.h
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoSceneView.h
    )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoGui.cpp
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoVideo.cpp
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoSceneView.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainGLFW.cpp
    )

# Group source files
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_executable(
        ${target}
        ${headers}
        ${sources}
    )

set_target_properties(
        ${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps"
    )

target_include_directories(
        ${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/
        ${SL_PROJECT_ROOT}/lib-SLProject/include
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/imgui
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/spa
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/dirent
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/glew/include
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/glfw3/include
        ${OpenCV_INCLUDE_DIR}
        PUBLIC
        INTERFACE
    )

target_link_libraries(
        ${target}
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
