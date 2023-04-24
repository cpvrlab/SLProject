# 
# CMake configuration for app-Demo-Node application
#

set(target app-Demo-Node)
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/app_demo_node/include/AppNodeGui.h
        ${SL_PROJECT_ROOT}/apps/app_demo_node/include/AppNodeSceneView.h
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.h
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
        ${SL_PROJECT_ROOT}/apps/source/SLScene.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/app_demo_node/source/AppNodeGui.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_node/source/AppNodeLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_node/source/AppNodeSceneView.cpp
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLScene.cpp
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
        ${SL_PROJECT_ROOT}/apps/app_demo_node/include
        ${SL_PROJECT_ROOT}/modules/sl/source
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui
        ${SL_PROJECT_ROOT}/modules/sl/externals/spa
        ${SL_PROJECT_ROOT}/modules/utils/externals/dirent
        ${SL_PROJECT_ROOT}/modules/sl/externals/gl3w
        ${SL_PROJECT_ROOT}/modules/sl/externals/gl3w/GL
        ${SL_PROJECT_ROOT}/apps/source
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
)

target_link_libraries(
        ${target}
        PRIVATE
        sl
        ${glfw_LIBS}
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

