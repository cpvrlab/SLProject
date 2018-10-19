# 
# CMake configuration for lib-SLExternal
#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(target lib-SLExternal)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}")
set(compile_definitions	GLEW_NO_GLU)
set(platform_specific_link_libraries)
set(platform_specific_include_directories)


file(GLOB headers
    ${include_path}/Shoemake/Decompose.h
    ${include_path}/Shoemake/EulerAngles.h
    ${include_path}/Shoemake/TypeDefs.h
    ${include_path}/imgui/imconfig.h
    ${include_path}/imgui/imgui_internal.h
    ${include_path}/imgui/imgui.h
    ${include_path}/imgui/stb_rect_pack.h
    ${include_path}/imgui/stb_textedit.h
    ${include_path}/imgui/stb_truetype.h
    ${include_path}/spa/spa.h
)

file(GLOB sources
    ${source_path}/Shoemake/EulerAngles.cpp
    ${source_path}/Shoemake/Decompose.cpp
    ${source_path}/imgui/imgui_draw.cpp
    ${source_path}/imgui/imgui.cpp
    ${source_path}/spa/spa.cpp
    )


#==============================================================================
if(NOT "${SYSTEM_NAME_UPPER}" STREQUAL "ANDROID")
    file(GLOB headers
        ${headers}
        ${include_path}/glew/include/GL/glew.h
        ${include_path}/glfw3/include/GLFW/glfw3.h
        ${include_path}/glfw3/include/GLFW/glfw3native.h
        ${include_path}/glfw3/src/glfw_config.h
        ${include_path}/glfw3/src/internal.h
        )

    file(GLOB sources
        ${sources}
        ${source_path}/glew/src/glew.c
        ${source_path}/glfw3/src/context.c
        ${source_path}/glfw3/src/init.c
        ${source_path}/glfw3/src/input.c
        ${source_path}/glfw3/src/monitor.c
        ${source_path}/glfw3/src/window.c
        )
endif()

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(compile_definitions
        ${compile_definitions}
        _GLFW_X11
        _GLFW_GLX
        _GLFW_USE_OPENGL
        _GLFW_HAS_GLXGETPROCADDRESS
        )

    file(GLOB headers
        ${headers}
        ${include_path}/glfw3/src/glfw_config.h
        ${include_path}/glfw3/src/x11_platform.h
        ${include_path}/glfw3/src/xkb_unicode.h
        ${include_path}/glfw3/src/linux_joystick.h
        ${include_path}/glfw3/src/posix_time.h
        ${include_path}/glfw3/src/posix_tls.h
        ${include_path}/glfw3/src/glx_context.h
        )

    file(GLOB sources
        ${sources}
        ${source_path}/glfw3/src/x11_init.c
        ${source_path}/glfw3/src/x11_monitor.c
        ${source_path}/glfw3/src/x11_window.c
        ${source_path}/glfw3/src/xkb_unicode.c
        ${source_path}/glfw3/src/linux_joystick.c
        ${source_path}/glfw3/src/posix_time.c
        ${source_path}/glfw3/src/posix_tls.c
        ${source_path}/glfw3/src/glx_context.c
        )

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    set(compile_definitions
        ${compile_definitions}
        GLEW_STATIC
        _GLFW_COCOA
        _GLFW_NSGL
        _GLFW_USE_OPENGL
        )

    file(GLOB headers
        ${headers}
        ${include_path}/glfw3/src/cocoa_platform.h
        ${include_path}/glfw3/src/iokit_joystick.h
        ${include_path}/glfw3/src/posix_tls.h
        ${include_path}/glfw3/src/nsgl_context.h
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
        ${source_path}/glfw3/src/mach_time.c
        ${source_path}/glfw3/src/posix_tls.c
        ${source_path}/glfw3/src/cocoa_init.m
        ${source_path}/glfw3/src/cocoa_monitor.m
        ${source_path}/glfw3/src/cocoa_window.m
        ${source_path}/glfw3/src/iokit_joystick.m
        ${source_path}/glfw3/src/nsgl_context.m
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
        GLEW_STATIC
        _GLFW_WIN32
        _GLFW_USE_OPENGL
        _GLFW_WGL
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
        )
	
    file(GLOB headers
        ${headers}
        ${include_path}/dirent/dirent.h
        ${include_path}/glew/include/GL/wglew.h
        ${include_path}/glfw3/src/win32_platform.h
        ${include_path}/glfw3/src/win32_tls.h
        ${include_path}/glfw3/src/winmm_joystick.h
        ${include_path}/glfw3/src/wgl_context.h
        ${include_path}/nvwa/debug_new.h
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
        ${source_path}/dirent/dirent.c
        ${source_path}/glfw3/src/win32_init.c
        ${source_path}/glfw3/src/win32_monitor.c
        ${source_path}/glfw3/src/win32_time.c
        ${source_path}/glfw3/src/win32_tls.c
        ${source_path}/glfw3/src/win32_window.c
        ${source_path}/glfw3/src/winmm_joystick.c
        ${source_path}/glfw3/src/wgl_context.c
        ${source_path}/nvwa/debug_new.cpp
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



#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #--------------------------------
    set(platform_specific_include_directories
        ${CMAKE_CURRENT_SOURCE_DIR}/dirent)
endif()
#==============================================================================

add_library(${target}
    ${headers}
    ${sources}
    )

add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "libs"
    )

target_include_directories(${target}
    PRIVATE

    PUBLIC
    ${SL_PROJECT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/half/include
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/src
    ${CMAKE_CURRENT_SOURCE_DIR}/nvwa
    ${CMAKE_CURRENT_SOURCE_DIR}/zlib
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/spa
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${platform_specific_include_directories}

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

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${PlatformLinkLibs}
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )

