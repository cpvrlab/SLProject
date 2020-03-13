# 
# CMake configuration for app-Demo-Node application
#

set(target app-Demo-Node)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/app-Demo-Node/include/AppNodeGui.h
        ${SL_PROJECT_ROOT}/apps/app-Demo-Node/include/AppNodeSceneView.h
        ${SL_PROJECT_ROOT}/apps/source/SLApplication.h
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.h
    )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/app-Demo-Node/source/AppNodeGui.cpp
        ${SL_PROJECT_ROOT}/apps/app-Demo-Node/source/AppNodeLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app-Demo-Node/source/AppNodeSceneView.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLApplication.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/AppNodeMainGLFW.cpp
    )

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
        ${SL_PROJECT_ROOT}/apps/app-Demo-Node/include
        ${SL_PROJECT_ROOT}/lib-SLProject/include
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/imgui
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/spa
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/dirent
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/glew/include
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/glfw3/include
        ${SL_PROJECT_ROOT}/apps/source
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

target_compile_definitions(
        ${target}
        PRIVATE
        ${compile_definitions}
        PUBLIC
        ${DEFAULT_COMPILE_DEFINITIONS}
        INTERFACE
    )

target_compile_options(
        ${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}
        INTERFACE
    )

target_link_libraries(
        ${target}
        PRIVATE
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        INTERFACE
    )

