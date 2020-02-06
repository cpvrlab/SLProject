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
)

set(headers
        ${headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Averaged.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AverageTiming.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FileLog.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FtpUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/HighResTimer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils.h
        ${UTILS_ROOT}/thirdparty/ftplibpp/ftplib.h
    )
set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/source/AverageTiming.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FileLog.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/FtpUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/Utils.cpp
        ${UTILS_ROOT}/thirdparty/ftplibpp/ftplib.cpp
    )
	
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
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${platform_specific_include_dirs}
    ${UTILS_ROOT}/thirdparty/asio/include
    ${UTILS_ROOT}/thirdparty/ftplibpp
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
