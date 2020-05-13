#
# CMake configuration for app-MapCreator-WAI application
#

set(target app-MapCreator-WAI)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

set(headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIAutoCalibration.h
    ${SL_PROJECT_ROOT}/apps/source/wai/AppWAISlamParamHelper.h
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.h
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSVideoStream.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSUtils.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/BRIEFPattern.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.h
    ${SL_PROJECT_ROOT}/apps/app-MapCreator-WAI/MapCreator.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app-MapCreator-WAI/main.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIAutoCalibration.cpp
    ${SL_PROJECT_ROOT}/apps/app-MapCreator-WAI/MapCreator.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.cpp
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSVideoStream.cpp
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSUtils.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.cpp
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
        ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/
        ${SL_PROJECT_ROOT}/apps/app-MapCreator-WAI/
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

    INTERFACE
    )
