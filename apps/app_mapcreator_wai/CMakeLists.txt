#
# CMake configuration for app-MapCreator-WAI application
#

set(target app-MapCreator-WAI)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

set(headers
    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/WAIAutoCalibration.h
    ${SL_PROJECT_ROOT}/apps/source/wai/AppWAISlamParamHelper.h
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.h
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.h
    ${SENS_ROOT}/SENSVideoStream.h
    ${SENS_ROOT}/SENSCalibration.h
    ${SENS_ROOT}/SENSUtils.h
    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLHessian.h
    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLextractor.h
    ${SL_PROJECT_ROOT}/apps/app_mapcreator_wai/MapCreator.h
    ${SL_PROJECT_ROOT}/apps/app_mapcreator_wai/WAIMapSlam.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app_mapcreator_wai/main.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/WAIAutoCalibration.cpp
    ${SL_PROJECT_ROOT}/apps/app_mapcreator_wai/MapCreator.cpp
    ${SL_PROJECT_ROOT}/apps/app_mapcreator_wai/WAIMapSlam.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.cpp
    ${SENS_ROOT}/SENSCalibration.cpp
    ${SENS_ROOT}/SENSVideoStream.cpp
    ${SENS_ROOT}/SENSUtils.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLHessian.cpp
    ${SL_PROJECT_ROOT}/apps/app_demo_wai/source/GLSLextractor.cpp
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
        ${SL_PROJECT_ROOT}/apps/app_mapcreator_wai/
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
