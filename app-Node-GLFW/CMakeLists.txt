# 
# CMake configuration for app-Node-GLFW application
# 

set(target app-Node-GLFW-exe)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/AppNodeGui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AppNodeSceneView.h
    )

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/AppNodeGui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppNodeSceneView.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppNodeMainGLFW.cpp
    )

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    )

target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/spa
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/dirent
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/glfw3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/lib-SLExternal/DBoW2
    ${OpenCV_INCLUDE_DIR}
    ${g2o_INCLUDE_DIR}
    PUBLIC
    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE
    lib-SLProject
    ${g2o_LIBS}
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
