#
# CMake project definition for exercise ch09_TextureMapping
#

set(target ch09_TextureMapping_Vulkan)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB headers
#        ${SL_PROJECT_ROOT}/lib-SLProject/include/glUtils.h
#        ${SL_PROJECT_ROOT}/lib-SLProject/include/CVTypedefs.h
#        ${SL_PROJECT_ROOT}/lib-SLProject/include/CVImage.h
		 ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.h
    )

file(GLOB sources
#        ${SL_PROJECT_ROOT}/lib-SLProject/source/glUtils.cpp
#        ${SL_PROJECT_ROOT}/lib-SLProject/source/SL/SL.cpp
#        ${SL_PROJECT_ROOT}/lib-SLProject/source/CV/CVImage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TextureMapping.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
    )

file(GLOB shaders
        ${SL_PROJECT_ROOT}/data/shaders/Diffuse.frag
        ${SL_PROJECT_ROOT}/data/shaders/Diffuse.vert
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
#    ${SL_PROJECT_ROOT}/lib-SLProject/include
#    ${SL_PROJECT_ROOT}/lib-SLProject/include/CV
#    ${SL_PROJECT_ROOT}/lib-SLProject/include/GL
    ${SL_PROJECT_ROOT}/lib-SLProject/include/math
#    ${SL_PROJECT_ROOT}/lib-SLProject/include/SL
#    ${SL_PROJECT_ROOT}/externals/lib-SLExternal
#    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/glew/include
    ${SL_PROJECT_ROOT}/externals/lib-SLExternal/glfw3/include
#    ${SL_PROJECT_ROOT}/externals/utils/lib-Utils/source
#    ${OpenCV_INCLUDE_DIR}
	${vk_INCLUDE_DIR}
    PUBLIC
    INTERFACE
    )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE
    ${PlatformLinkLibs}
    lib-SLExternal
#    lib-Utils
#    ${OpenCV_LIBS}
	${vk_LIBS}
    PUBLIC
    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE
    ${compile_definitions}
    GLEW_STATIC
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

