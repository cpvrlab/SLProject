#
# CMake configuration for lib-SLMath
#

set(target lib-SLMath)

file(GLOB headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLAlgo.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLCurve.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLMat3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLMat4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLMath.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLPlane.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLQuat4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLVec2.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLVec3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLVec4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SLVector.h

        ${CMAKE_CURRENT_SOURCE_DIR}/externals/Shoemake/Decompose.h
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/Shoemake/EulerAngles.h
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/Shoemake/TypeDefs.h
        )

set(sources
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLAlgo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SLPlane.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/externals/Shoemake/EulerAngles.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/Shoemake/Decompose.cpp
        )

# Group source files
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/include" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/source" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

#==============================================================================
add_library(${target}
            ${headers}
            ${sources})

add_library(${META_PROJECT_NAME}::${target}
        ALIAS
        ${target}
        )

set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

target_include_directories(${target}
        PRIVATE
        ${SL_PROJECT_ROOT}/externals/libigl/include
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${SL_PROJECT_ROOT}/externals/eigen
		${CMAKE_CURRENT_SOURCE_DIR}/externals
        ${DEFAULT_INCLUDE_DIRECTORIES}

        INTERFACE
        )

target_compile_definitions(${target}
        PRIVATE

        PUBLIC
        ${compile_definitions}
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
        ${PlatformLinkLibs}
        ${META_PROJECT_NAME}::lib-SLExternal
        lib-Utils
        ${DEFAULT_LINKER_OPTIONS}

        INTERFACE

        )
