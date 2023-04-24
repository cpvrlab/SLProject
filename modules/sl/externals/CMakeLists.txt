# 
# CMake configuration for libsl_external
#

set(target sl_external)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(platform_specific_link_libraries)
set(platform_specific_include_directories)

set(headers
        ${include_path}/imgui/imconfig.h
        ${include_path}/imgui/imgui_internal.h
        ${include_path}/imgui/imgui.h
        ${include_path}/imgui/imstb_rectpack.h
        ${include_path}/imgui/imstb_textedit.h
        ${include_path}/imgui/imstb_truetype.h
        ${include_path}/imgui-addons/bezier.hpp
        ${include_path}/imgui-addons/imgui_color_gradient.h
        ${include_path}/spa/spa.h
        )

set(sources
        ${include_path}/imgui/imgui.cpp
        ${include_path}/imgui/imgui_demo.cpp
        ${include_path}/imgui/imgui_draw.cpp
        ${include_path}/imgui/imgui_tables.cpp
        ${include_path}/imgui/imgui_widgets.cpp
        ${include_path}/imgui-addons/bezier.cpp
        ${include_path}/imgui-addons/imgui_color_gradient.cpp
        ${include_path}/spa/spa.cpp
        )

#==============================================================================
if(NOT "${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID")
    file(GLOB headers
            ${headers}
            )

    file(GLOB sources
            ${sources}
            )
endif()

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(compile_definitions
            ${compile_definitions}
            )

    file(GLOB headers
            ${headers}
            ${include_path}/gl3w/GL/gl3w.h
            ${include_path}/gl3w/GL/glcorearb.h
            )

    file(GLOB sources
            ${sources}
            ${source_path}/gl3w/GL/gl3w.c
            )

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    set(compile_definitions
            ${compile_definitions}
            )

    file(GLOB headers
            ${headers}
            ${include_path}/gl3w/GL/gl3w.h
            ${include_path}/gl3w/GL/glcorearb.h
            ${include_path}/zlib/ioapi.h
            ${include_path}/zlib/unzip.h
            ${include_path}/zlib/zconf.in.h
            ${include_path}/zlib/zutil.h
            ${include_path}/zlib/zlib.h
            ${include_path}/zlib/zconf.h
            ${include_path}/zlib/trees.h
            ${include_path}/zlib/inftrees.h
            ${include_path}/zlib/inflate.h
            ${include_path}/zlib/inffixed.h
            ${include_path}/zlib/inffast.h
            ${include_path}/zlib/gzguts.h
            ${include_path}/zlib/deflate.h
            ${include_path}/zlib/crypt.h
            ${include_path}/zlib/crc32.h
        )

    file(GLOB sources
            ${sources}
            ${source_path}/gl3w/GL/gl3w.c
            ${source_path}/zlib/zutil.c
            ${source_path}/zlib/uncompr.c
            ${source_path}/zlib/trees.c
            ${source_path}/zlib/inftrees.c
            ${source_path}/zlib/inflate.c
            ${source_path}/zlib/inffast.c
            ${source_path}/zlib/infback.c
            ${source_path}/zlib/gzwrite.c
            ${source_path}/zlib/gzread.c
            ${source_path}/zlib/gzlib.c
            ${source_path}/zlib/gzclose.c
            ${source_path}/zlib/deflate.c
            ${source_path}/zlib/crc32.c
            ${source_path}/zlib/compress.c
            ${source_path}/zlib/adler32.c
            ${source_path}/zlib/ioapi.c
            ${source_path}/zlib/unzip.c
        )

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(compile_definitions
            ${compile_definitions}
            _CRT_SECURE_NO_WARNINGS
            UNICODE
            _UNICODE
            )
	
    file(GLOB headers
            ${headers}
            ${include_path}/gl3w/GL/gl3w.h
            ${include_path}/gl3w/GL/glcorearb.h
            ${include_path}/zlib/ioapi.h
            ${include_path}/zlib/unzip.h
            ${include_path}/zlib/zconf.in.h
            ${include_path}/zlib/zutil.h
            ${include_path}/zlib/zlib.h
            ${include_path}/zlib/zconf.h
            ${include_path}/zlib/trees.h
            ${include_path}/zlib/inftrees.h
            ${include_path}/zlib/inflate.h
            ${include_path}/zlib/inffixed.h
            ${include_path}/zlib/inffast.h
            ${include_path}/zlib/gzguts.h
            ${include_path}/zlib/deflate.h
            ${include_path}/zlib/crypt.h
            ${include_path}/zlib/crc32.h
            )
  
    file(GLOB sources
            ${sources}
            ${source_path}/gl3w/GL/gl3w.c
            ${source_path}/zlib/zutil.c
            ${source_path}/zlib/uncompr.c
            ${source_path}/zlib/trees.c
            ${source_path}/zlib/inftrees.c
            ${source_path}/zlib/inflate.c
            ${source_path}/zlib/inffast.c
            ${source_path}/zlib/infback.c
            ${source_path}/zlib/gzwrite.c
            ${source_path}/zlib/gzread.c
            ${source_path}/zlib/gzlib.c
            ${source_path}/zlib/gzclose.c
            ${source_path}/zlib/deflate.c
            ${source_path}/zlib/crc32.c
            ${source_path}/zlib/compress.c
            ${source_path}/zlib/adler32.c
            ${source_path}/zlib/ioapi.c
            ${source_path}/zlib/unzip.c
            )

endif()
#==============================================================================

# Group source files (for visual studio ide)
set(headers_sources ${headers} ${sources})
# combination of headers and sources as the sub libs already have source and header directories
source_group_by_path("${include_path}" "\\\\.h$|\\\\.hpp$" "" ${headers_sources})

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
        ${SL_PROJECT_ROOT}/modules/sl/source
        ${CMAKE_CURRENT_SOURCE_DIR}/gl3w
        ${CMAKE_CURRENT_SOURCE_DIR}/gl3w/GL
        ${CMAKE_CURRENT_SOURCE_DIR}/half/include
        ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/src
        ${CMAKE_CURRENT_SOURCE_DIR}/nvwa
        ${CMAKE_CURRENT_SOURCE_DIR}/zlib
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui-addons
        ${CMAKE_CURRENT_SOURCE_DIR}/spa
        ${CMAKE_CURRENT_SOURCE_DIR}/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${platform_specific_include_directories}
        ${OpenCV_INCLUDE_DIR}
        INTERFACE
        )

target_compile_definitions(${target}
        PRIVATE
        ${compile_definitions}
        PUBLIC
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
