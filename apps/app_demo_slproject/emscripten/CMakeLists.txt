set(target app-Demo-SLProject)

file(GLOB headers
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.h
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoSceneView.h
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoGui.h
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoVideo.h
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
        ${SL_PROJECT_ROOT}/apps/source/Scene.h
        ${SL_PROJECT_ROOT}/apps/source/CVCapture.h
        ${SL_PROJECT_ROOT}/apps/source/WebCamera.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoSceneView.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoGui.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoLoad.cpp
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoVideo.cpp
        ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
        ${SL_PROJECT_ROOT}/apps/source/CVCapture.cpp
        ${SL_PROJECT_ROOT}/apps/source/WebCamera.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainEmscripten.cpp
        )

add_executable(
        ${target}
        ${headers}
        ${sources}
)

target_include_directories(
        ${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/
        ${SL_PROJECT_ROOT}/apps/source
        ${SL_PROJECT_ROOT}/modules/sl/source/accelstruct
        ${SL_PROJECT_ROOT}/modules/sl/source/animation
        ${SL_PROJECT_ROOT}/modules/sl/source/cv
        ${SL_PROJECT_ROOT}/modules/sl/source/gl
        ${SL_PROJECT_ROOT}/modules/sl/source/input
        ${SL_PROJECT_ROOT}/modules/sl/source/mesh
        ${SL_PROJECT_ROOT}/modules/sl/source/node
        ${SL_PROJECT_ROOT}/modules/sl/source/optix
        ${SL_PROJECT_ROOT}/modules/sl/source/ray
        ${SL_PROJECT_ROOT}/modules/sl/source
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

# Copy index.html and server script to output directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/index.html" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.html" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/server.py" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/server.py" COPYONLY)
