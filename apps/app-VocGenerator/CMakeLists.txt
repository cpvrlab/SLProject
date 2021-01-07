#
# CMake configuration for app-VocGenerator application
#

set(target app-VocGenerator)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

set(headers
    ${SL_PROJECT_ROOT}/apps/source/wai/AppWAISlamParamHelper.h
    ${SENS_ROOT}/SENSUtils.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app-VocGenerator/VocGenerator.cpp
    ${SENS_ROOT}/SENSUtils.cpp
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
        ${SL_PROJECT_ROOT}/apps/source/
        ${SL_PROJECT_ROOT}/apps/source/wai
        ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/
        ${SL_PROJECT_ROOT}/apps/app-VocGenerator/
        ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
	    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source
	    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/Gui
	    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GuiPrefab
        ${SL_PROJECT_ROOT}/modules/sl/externals/gl3w
		${SL_PROJECT_ROOT}/modules/sl/externals/gl3w/GL
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC

        INTERFACE
    )

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        lib-SLProject
        lib-WAI
        lib-fbow
        lib-DBoW2
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
    ${META_PROJECT_NAME}::lib-Utils

    INTERFACE
    )
