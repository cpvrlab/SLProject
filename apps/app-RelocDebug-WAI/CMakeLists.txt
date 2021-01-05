#
# CMake configuration for app-RelocDebug-WAI application
#

set(target app-RelocDebug)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

set(headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIAutoCalibration.h
    ${SL_PROJECT_ROOT}/apps/source/wai/AppWAISlamParamHelper.h
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.h
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.h
    ${SENS_ROOT}/SENSVideoStream.h
    ${SENS_ROOT}/SENSCalibration.h
    ${SENS_ROOT}/SENSUtils.h
	${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.h
	${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.h
    )

set(sources
    ${SL_PROJECT_ROOT}/apps/app-RelocDebug-WAI/main.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIAutoCalibration.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.cpp
    ${SENS_ROOT}/SENSCalibration.cpp
    ${SENS_ROOT}/SENSVideoStream.cpp
    ${SENS_ROOT}/SENSUtils.cpp
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
	    ${SENS_ROOT}
        ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/
        ${SL_PROJECT_ROOT}/apps/app-RelocDebug-WAI/
        ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
	    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source
	    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/Gui
	    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GuiPrefab
        ${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w
		${SL_PROJECT_ROOT}/externals/lib-SLExternal/gl3w/GL
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
