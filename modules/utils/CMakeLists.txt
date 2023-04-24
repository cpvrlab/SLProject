#
# CMake configuration of the cpvrlab/utils repository
#

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# The FOLDER properties are needed in Visual Studio and XCode generated projects for nested folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

set(UTILS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

include(${SL_PROJECT_ROOT}/cmake/CompileOptions.cmake)
include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

if(NOT "${SYSTEM_NAME_UPPER}" STREQUAL "IOS" AND NOT "${SYSTEM_NAME_UPPER}" STREQUAL "EMSCRIPTEN")
	add_subdirectory(externals/zlib)
endif()

add_subdirectory(tests)

#
# CMake configuration for libsl_utils
#

set(target sl_utils)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(platform_specific_link_libraries)
set(platform_specific_include_directories)

set(compile_definitions
    ${compile_definitions}
    ASIO_STANDALONE
    ASIO_HAS_STD_TYPE_TRAITS
    ASIO_HAS_STD_SHARED_PTR
    ASIO_HAS_STD_ADDRESSOF
    NOSSL
    NOLFS
    _CRT_NO_VA_START_VALIDATION
    WIN32_LEAN_AND_MEAN
)

set(headers
        ${headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Averaged.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AverageTiming.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FileLog.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FtpUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/HighResTimer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GlobalTimer.h
	    ${CMAKE_CURRENT_SOURCE_DIR}/source/CustomLog.h
#		${CMAKE_CURRENT_SOURCE_DIR}/source/Instrumentor.h
		${CMAKE_CURRENT_SOURCE_DIR}/source/Profiler.h
		${CMAKE_CURRENT_SOURCE_DIR}/source/ZipUtils.h
		${CMAKE_CURRENT_SOURCE_DIR}/source/HttpUtils.h
		${CMAKE_CURRENT_SOURCE_DIR}/source/Utils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLFileStorage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOBrowserPopup.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOFetch.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOLocalStorage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOMemory.h
#		${CMAKE_CURRENT_SOURCE_DIR}/source/CoordSystemConversions.h
		${UTILS_ROOT}/externals/ftplibpp/ftplib.h
    )
set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIONative.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AverageTiming.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FileLog.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FtpUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GlobalTimer.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/source/Profiler.cpp
	    ${CMAKE_CURRENT_SOURCE_DIR}/source/ZipUtils.cpp
	    ${CMAKE_CURRENT_SOURCE_DIR}/source/HttpUtils.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/source/ByteOrder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLFileStorage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOBrowserPopup.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOFetch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOLocalStorage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIOMemory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/io/SLIONative.cpp
    )

# Emscripten provides its own implementation of zlib
#==============================================================================
if(NOT "${SYSTEM_NAME_UPPER}" STREQUAL "EMSCRIPTEN")
    set(sources
        ${sources}
        ${UTILS_ROOT}/externals/ftplibpp/ftplib.cpp
        ${UTILS_ROOT}/externals/zlib/contrib/minizip/ioapi.c
        ${UTILS_ROOT}/externals/zlib/contrib/minizip/unzip.c
        ${UTILS_ROOT}/externals/zlib/contrib/minizip/zip.c
    )
endif()
#==============================================================================

if("${SYSTEM_NAME_UPPER}" STREQUAL "IOS")
    set(headers
        ${headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils_iOS.h
        )
    set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils_iOS.mm
        )
endif()

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS")
    set(platform_specific_include_dirs
        ${UTILS_ROOT}/externals/dirent)
    set(headers ${headers}
        ${UTILS_ROOT}/externals/dirent/dirent.h)
    set(sources ${sources}
        ${UTILS_ROOT}/externals/dirent/dirent.c)
endif()
#==============================================================================

# Group source files (for visual studio ide)
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/source" "\\\\.h$|\\\\.hpp$" "Header Files" ${headers})
source_group_by_path("${CMAKE_CURRENT_SOURCE_DIR}/source" "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" "Source Files" ${sources})

add_library(${target}
    ${headers}
    ${sources}
    )

add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "external"
    )

target_include_directories(${target}
    PRIVATE

    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/source/io
    ${platform_specific_include_dirs}
    ${UTILS_ROOT}/externals/asio/include
    ${UTILS_ROOT}/externals/ftplibpp
	${UTILS_ROOT}/externals/zlib/contrib
    ${openssl_INCLUDE_DIR}
    INTERFACE
    )

if (NOT "${SYSTEM_NAME_UPPER}" STREQUAL "EMSCRIPTEN")
    target_include_directories(${target}
        PRIVATE
        PUBLIC
        ${UTILS_ROOT}/externals/zlib
        INTERFACE
    )
endif()

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
    ${EXTERNAL_LIB_COMPILE_OPTIONS}

    INTERFACE
    )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${PlatformLinkLibs}
    ${DEFAULT_LINKER_OPTIONS}
    ${openssl_LIBS}

    INTERFACE
    )

# Emscripten provides its own implementation of zlib
if (NOT "${SYSTEM_NAME_UPPER}" STREQUAL "EMSCRIPTEN")
    target_link_libraries(${target} PUBLIC zlib)
else()
    target_compile_options(${target} PUBLIC "-sUSE_ZLIB=1")
    target_link_options(${target} PUBLIC "-sUSE_ZLIB=1")
endif()