# 
# CMake configuration for app-Demo-SLProject application
#

set(target app-Demo-SLProject)

set(compile_definitions SL_STARTSCENE=SID_ShaderIBL)

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoGui.h
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoSceneView.h
		${SL_PROJECT_ROOT}/apps/source/CVCapture.h
		${SL_PROJECT_ROOT}/apps/source/AppDemo.h
		${SL_PROJECT_ROOT}/apps/source/SLInterface.h
        ${SL_PROJECT_ROOT}/apps/source/Scene.h
    )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoGui.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoVideo.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoSceneView.cpp
		${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
		${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
		${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.cpp
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
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/
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
        ${PlatformLinkLibs}
        sl_external
        sl_utils
        ${OpenCV_LIBS}
        ${glfw_LIBS}
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
