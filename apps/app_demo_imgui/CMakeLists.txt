#
# CMake project definition for Imgui demo app
#

set(target app-Demo-Imgui)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imconfig.h
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui.h
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui_internal.h
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui_glfw.h
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/backends/imgui_impl_opengl3.h
    )

file(GLOB sources
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui.cpp
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui_demo.cpp
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui_draw.cpp
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/imgui_widgets.cpp
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/backends/imgui_impl_glfw.cpp
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/backends/imgui_impl_opengl3.cpp
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/examples/example_glfw_opengl3/main.cpp
    )

add_executable(${target}
        ${headers}
        ${sources}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps"
        ENABLE_EXPORTS ON
        )

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/modules/sl/externals
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/backends
        ${SL_PROJECT_ROOT}/modules/sl/externals/imgui/examples
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
        )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
        PRIVATE
        ${PlatformLinkLibs}
        sl_external
        ${glfw_LIBS}
        PUBLIC
        INTERFACE
        )

target_compile_definitions(${target}
        PRIVATE
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
