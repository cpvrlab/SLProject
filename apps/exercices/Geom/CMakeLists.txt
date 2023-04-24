#
# CMake project definition for Geom
#

set(target Geom)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
        ${SL_PROJECT_ROOT}/modules/sl/source/gl/glUtils.h
        ${SL_PROJECT_ROOT}/modules/sl/source/cv/CVTypedefs.h
        ${SL_PROJECT_ROOT}/modules/sl/source/cv/CVImage.h
        )

file(GLOB sources
        ${SL_PROJECT_ROOT}/modules/sl/source/gl/glUtils.cpp
        ${SL_PROJECT_ROOT}/modules/sl/source/sl/SL.cpp
        ${SL_PROJECT_ROOT}/modules/sl/source/cv/CVImage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Geom.cpp
        )

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/Geom.vert
        ${SL_PROJECT_ROOT}/data/shaders/Geom.frag
	${SL_PROJECT_ROOT}/data/shaders/Geom.geom
        )

add_executable(${target}
        ${headers}
        ${sources}
        ${shaders}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        FOLDER "apps/exercices"
        )

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/modules/math/source
        ${SL_PROJECT_ROOT}/modules/math/externals
        ${SL_PROJECT_ROOT}/modules/sl/source
        ${SL_PROJECT_ROOT}/modules/sl/source/cv
        ${SL_PROJECT_ROOT}/modules/sl/source/gl
        ${SL_PROJECT_ROOT}/modules/sl/source/sl
        ${SL_PROJECT_ROOT}/modules/sl/externals
        ${SL_PROJECT_ROOT}/modules/sl/externals/gl3w
        ${SL_PROJECT_ROOT}/modules/sl/externals/gl3w/GL
        ${SL_PROJECT_ROOT}/modules/utils/source
        ${OpenCV_INCLUDE_DIR}
        ${glfw_INCLUDE_DIR}
        PUBLIC
        INTERFACE
        )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
        PRIVATE
        ${PlatformLinkLibs}
        sl_external
        sl_utils
        ${OpenCV_LIBS}
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

