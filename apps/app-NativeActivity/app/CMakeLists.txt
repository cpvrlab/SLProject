# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html
#find_package(OpenCV REQUIRED)

set(target native-lib)

file(GLOB headers
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeCameraInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeSensorsInterface.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeWAIMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeCameraInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/AppDemoNativeSensorsInterface.cpp
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
    ${ANDROID_NDK}/sources/android/native_app_glue/
    ${CMAKE_CURRENT_SOURCE_DIR}//src/main/cpp/
    PUBLIC
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

    PUBLIC
        libc++_shared.so
        z

    INTERFACE
    )
