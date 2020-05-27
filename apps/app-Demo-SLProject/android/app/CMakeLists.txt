# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(target native-lib)

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoAndroidJNI.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoGui.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoLoad.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoVideo.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/source/AppDemoSceneView.cpp
    ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.cpp
    ${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
    ${SL_PROJECT_ROOT}/apps/source/SLApplication.cpp
    ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
 )

set(headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoGui.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include/AppDemoSceneView.h
    ${SL_PROJECT_ROOT}/apps/source/SLProjectScene.h
    ${SL_PROJECT_ROOT}/apps/source/CVCapture.h
    ${SL_PROJECT_ROOT}/apps/source/SLApplication.h
    ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
    )

add_library(${target}
    SHARED
    ${sources}
    ${headers}
    )

target_include_directories(
    ${target}
    PRIVATE
    PUBLIC
    ${SL_PROJECT_ROOT}/apps/app-Demo-SLProject/include
    ${SL_PROJECT_ROOT}/apps/source
    ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
    ${SL_PROJECT_ROOT}/lib-SLProject/include
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/imgui
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/spa
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/zlib
    ${OpenCV_INCLUDE_DIR}
    ${assimp_INCLUDE_DIR}
    ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(
        ${target}
        ${DEFAULT_LIBRARIES}
        libc++_shared.so
        ${OpenCV_LIBS}
        ${assimp_LIBS}
        ${META_PROJECT_NAME}::lib-SLProject
    )

#
# Copy APK contents
#
include(CopyResourcesAppDemoSLProject)
copy_resources_slprojectdemo("${CMAKE_CURRENT_LIST_DIR}/src/main/assets")