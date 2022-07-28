set(target app-Demo-SLProject)

file(GLOB headers
    ${SL_PROJECT_ROOT}/apps/source/AppDemo.h
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoSceneView.h
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/include/AppDemoGui.h
    ${SL_PROJECT_ROOT}/apps/source/SLInterface.h
    ${SL_PROJECT_ROOT}/apps/source/Scene.h
)

file(GLOB sources
    ${SL_PROJECT_ROOT}/apps/source/AppDemo.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoSceneView.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_slproject/source/AppDemoGui.cpp
    ${SL_PROJECT_ROOT}/apps/source/SLInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainWASM.cpp
)

add_executable(
    ${target}
    ${headers}
    ${sources}
)

# Link GLFW 3
set_target_properties(${target} PROPERTIES LINK_FLAGS "-s USE_GLFW=3")

# Copy index.html to output directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/index.html" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.html" COPYONLY)

message(${OpenCV_INCLUDE_DIR})

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
        lib-SLProject
        ${PlatformLinkLibs}
        lib-SLExternal
        lib-Utils
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
