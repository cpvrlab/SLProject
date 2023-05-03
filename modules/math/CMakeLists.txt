#
# CMake configuration for sl_math
#

set(target sl_math)

file(GLOB headers
        source/SLAlgo.h
        source/SLCurve.h
        source/SLMat3.h
        source/SLMat4.h
        source/SLMath.h
        source/SLPlane.h
        source/SLQuat4.h
        source/SLVec2.h
        source/SLVec3.h
        source/SLVec4.h
        source/SLVector.h

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

        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/source
	${CMAKE_CURRENT_SOURCE_DIR}/externals
        ${SL_PROJECT_ROOT}/externals/eigen
        ${SL_PROJECT_ROOT}/externals/libigl/include
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
        sl_external
        sl_utils
        ${DEFAULT_LINKER_OPTIONS}

        INTERFACE

        )
