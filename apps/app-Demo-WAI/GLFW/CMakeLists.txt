#
# CMake configuration for app-Demo-WAI application
#

set(target app-Demo-WAI)

set(compile_definitions
    SL_STARTSCENE=SID_Minimal)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")


set(headers
    ${erlebar_headers}
    ${SL_PROJECT_ROOT}/apps/source/sens/GLFW/SENSWebCamera.h
    )

set(sources 
    ${erlebar_sources}
    ${SL_PROJECT_ROOT}/apps/source/sens/GLFW/SENSWebCamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AppDemoMainGLFW.cpp
    )

# Group source files
set(headers_sources ${sources} ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "" ${headers_sources})
#source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
#source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_executable(${target}
    ${headers}
    ${sources}
    )

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "apps"
    )

target_include_directories(${target}
    PRIVATE
    ${erlebar_include_directories}
    PUBLIC
    ${glfw_INCLUDE_DIR}
    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${META_PROJECT_NAME}::lib-WAI
    ${META_PROJECT_NAME}::lib-fbow
    ${META_PROJECT_NAME}::lib-DBoW2
    lib-SLProject
    ${glfw_LIBS}

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}

    PUBLIC
    ${DEFAULT_COMPILE_DEFINITIONS}
    WAI_DLL
    WAI_DLL_IMPORTS

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
