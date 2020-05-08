#
# CMake configuration for app-Test-WAI application
#

set(target app-Test-WAI)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIAutoCalibration.h
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.h
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.h
    ${SL_PROJECT_ROOT}/apps/app-Test-WAI/WAITester.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/BRIEFPattern.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.h
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENS.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSCamera.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSFrame.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSException.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSUtils.h
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSVideoStream.h
    ${SL_PROJECT_ROOT}/apps/source/sens/GLFW/SENSWebCamera.h
    )

file(GLOB sources
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/WAIAutoCalibration.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/WAIMapStorage.cpp
    ${SL_PROJECT_ROOT}/apps/source/wai/FeatureExtractorFactory.cpp
    ${SL_PROJECT_ROOT}/apps/app-Test-WAI/WAITester.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLHessian.cpp
    ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/GLSLextractor.cpp
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSCamera.cpp
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSUtils.cpp
    ${SL_PROJECT_ROOT}/apps/source/sens/SENSVideoStream.cpp
    ${SL_PROJECT_ROOT}/apps/source/sens/GLFW/SENSWebCamera.cpp
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
        ${SL_PROJECT_ROOT}/apps/app-Test-WAI/
        ${SL_PROJECT_ROOT}/apps/app-Demo-WAI/source/
        ${SL_PROJECT_ROOT}/apps/source/wai/
        ${SL_PROJECT_ROOT}/apps/source/sens/
        ${SL_PROJECT_ROOT}/apps/source/
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${META_PROJECT_NAME}::lib-WAI
        ${META_PROJECT_NAME}::lib-fbow
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
