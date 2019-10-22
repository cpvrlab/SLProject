#
# CMake configuration for app-MapCreator-WAI application
#

set(target app-MapCreator-WAI)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

set(headers
    ${SL_PROJECT_ROOT}/apps/source/wai/AppWaiSlamParamHelper.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app-MapCreator-WAI/main.cpp
    )

# Group source files
set(headers_sources ${sources} ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}" "\\\\.h$|\\\\.hpp$" "" ${headers_sources})

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
    ${SL_PROJECT_ROOT}/apps/source/wai
    PUBLIC

    INTERFACE
    )

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    lib-SLProject
    lib-WAI

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE

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
