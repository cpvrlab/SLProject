# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

set(target native-lib)

set(headers
		${erlebar_headers}
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCamera.h
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCameraUtils.h
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkGps.h
    )

set(sources
		${erlebar_sources}
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCamera.cpp
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkCameraUtils.cpp
		${SL_PROJECT_ROOT}/apps/source/sens/android/SENSNdkGps.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeWAIMain.cpp
    )

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_library(${target}
    SHARED
        ${sources}
        ${headers}
    )

target_include_directories(${target}
    PRIVATE
		${erlebar_include_directories}
        ${ANDROID_NDK}/sources/android/native_app_glue/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/
    PUBLIC
        ${DEFAULT_INCLUDE_DIRECTORIES}
    )

target_link_libraries(${target}
    PRIVATE
        android
        native_app_glue
        EGL
        GLESv3
        log
        camera2ndk
        mediandk
        lib-WAI
        lib-DBoW2
        lib-SLProject

    PUBLIC
        libc++_shared.so
        z

    INTERFACE
    )

#
# Copy APK contents
#
include(CopyResourcesErlebAR)
copy_resources_erlebar("${CMAKE_CURRENT_LIST_DIR}/src/main/assets/data")
#does not work because of size of erleb-AR directory
#copy_resources_erlebar_test("${CMAKE_CURRENT_LIST_DIR}/src/main/assets/erleb-AR")