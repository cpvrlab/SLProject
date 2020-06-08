# 
# CMake configuration for lib-Utils
#

set(target lib-Utils)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Instrumentor.h
		${CMAKE_CURRENT_SOURCE_DIR}/source/ZipUtils.h
        ${UTILS_ROOT}/thirdparty/ftplibpp/ftplib.h
        ${UTILS_ROOT}/thirdparty/zlib/zconf.h.included
        ${UTILS_ROOT}/thirdparty/zlib/zlib.h
    )
set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AverageTiming.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FileLog.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FtpUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/GlobalTimer.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/source/ZipUtils.cpp
        ${UTILS_ROOT}/thirdparty/ftplibpp/ftplib.cpp
        ${UTILS_ROOT}/thirdparty/zlib/adler32.c
        ${UTILS_ROOT}/thirdparty/zlib/compress.c
        ${UTILS_ROOT}/thirdparty/zlib/crc32.c
        ${UTILS_ROOT}/thirdparty/zlib/deflate.c
        ${UTILS_ROOT}/thirdparty/zlib/gzclose.c
        ${UTILS_ROOT}/thirdparty/zlib/gzlib.c
        ${UTILS_ROOT}/thirdparty/zlib/gzread.c
        ${UTILS_ROOT}/thirdparty/zlib/gzwrite.c
        ${UTILS_ROOT}/thirdparty/zlib/inflate.c
        ${UTILS_ROOT}/thirdparty/zlib/infback.c
        ${UTILS_ROOT}/thirdparty/zlib/inflate.c
        ${UTILS_ROOT}/thirdparty/zlib/inftrees.c
        ${UTILS_ROOT}/thirdparty/zlib/inffast.c
        ${UTILS_ROOT}/thirdparty/zlib/trees.c
        ${UTILS_ROOT}/thirdparty/zlib/uncompr.c
        ${UTILS_ROOT}/thirdparty/zlib/zutil.c
        ${UTILS_ROOT}/thirdparty/zlib/contrib/minizip/ioapi.c
        ${UTILS_ROOT}/thirdparty/zlib/contrib/minizip/unzip.c
        ${UTILS_ROOT}/thirdparty/zlib/contrib/minizip/zip.c
    )
    
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
        ${UTILS_ROOT}/thirdparty/dirent)
    set(headers ${headers}
        ${UTILS_ROOT}/thirdparty/dirent/dirent.h)
    set(sources ${sources}
        ${UTILS_ROOT}/thirdparty/dirent/dirent.c)
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
<<<<<<< HEAD
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${platform_specific_include_dirs}
    ${UTILS_ROOT}/thirdparty/asio/include
    ${UTILS_ROOT}/thirdparty/ftplibpp
    ${UTILS_ROOT}/thirdparty/zlib
    ${UTILS_ROOT}/thirdparty/zlib/contrib
    ${DEFAULT_INCLUDE_DIRECTORIES}
=======
        ${CMAKE_CURRENT_SOURCE_DIR}/source
        ${platform_specific_include_dirs}
        ${UTILS_ROOT}/thirdparty/asio/include
        ${UTILS_ROOT}/thirdparty/ftplibpp
        ${DEFAULT_INCLUDE_DIRECTORIES}
>>>>>>> fcf62d327286690708f4b21c326aa95e7ba093a0

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
    ${EXTERNAL_LIB_COMPILE_OPTIONS}

    INTERFACE
    )

include(${SL_PROJECT_ROOT}/cmake/PlatformLinkLibs.cmake)

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${PlatformLinkLibs}
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )
