# 
# CMake configuration for lib-SLExternal
#

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
    ${include_path}/Shoemake/EulerAngles.cpp
    ${include_path}/Shoemake/Decompose.cpp
    ${include_path}/imgui/imgui_draw.cpp
    ${include_path}/imgui/imgui.cpp
    ${include_path}/spa/spa.cpp
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
        ${include_path}/glew/src/glew.c
        ${include_path}/glfw3/src/context.c
        ${include_path}/glfw3/src/init.c
        ${include_path}/glfw3/src/input.c
        ${include_path}/glfw3/src/monitor.c
        ${include_path}/glfw3/src/window.c
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
        ${include_path}/glfw3/src/x11_init.c
        ${include_path}/glfw3/src/x11_monitor.c
        ${include_path}/glfw3/src/x11_window.c
        ${include_path}/glfw3/src/xkb_unicode.c
        ${include_path}/glfw3/src/linux_joystick.c
        ${include_path}/glfw3/src/posix_time.c
        ${include_path}/glfw3/src/posix_tls.c
        ${include_path}/glfw3/src/glx_context.c
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
        ${include_path}/zlib/dirent.h
        )

    file(GLOB sources
        ${sources}
        ${include_path}/glfw3/src/mach_time.c
        ${include_path}/glfw3/src/posix_tls.c
        ${include_path}/glfw3/src/cocoa_init.m
        ${include_path}/glfw3/src/cocoa_monitor.m
        ${include_path}/glfw3/src/cocoa_window.m
        ${include_path}/glfw3/src/iokit_joystick.m
        ${include_path}/glfw3/src/nsgl_context.m
        ${include_path}/zlib/zutil.c
        ${include_path}/zlib/uncompr.c
        ${include_path}/zlib/trees.c
        ${include_path}/zlib/inftrees.c
        ${include_path}/zlib/inflate.c
        ${include_path}/zlib/inffast.c
        ${include_path}/zlib/infback.c
        ${include_path}/zlib/gzwrite.c
        ${include_path}/zlib/gzread.c
        ${include_path}/zlib/gzlib.c
        ${include_path}/zlib/gzclose.c
        ${include_path}/zlib/deflate.c
        ${include_path}/zlib/crc32.c
        ${include_path}/zlib/compress.c
        ${include_path}/zlib/adler32.c
        ${include_path}/zlib/ioapi.c
        ${include_path}/zlib/unzip.c
        ${include_path}/zlib/dirent.c
        )

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(compile_definitions
        ${compile_definitions}
        GLEW_STATIC
        _GLFW_WIN32
        _GLFW_USE_OPENGL
        _GLFW_WGL
        UNICODE
        _UNICODE
        )
	
    file(GLOB headers
        ${headers}
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
        ${include_path}/zlib/dirent.h
        )
  
    file(GLOB sources
        ${sources}
        ${include_path}/glfw3/src/win32_init.c
        ${include_path}/glfw3/src/win32_monitor.c
        ${include_path}/glfw3/src/win32_time.c
        ${include_path}/glfw3/src/win32_tls.c
        ${include_path}/glfw3/src/win32_window.c
        ${include_path}/glfw3/src/winmm_joystick.c
        ${include_path}/glfw3/src/wgl_context.c
        ${include_path}/nvwa/debug_new.cpp
        ${include_path}/zlib/zutil.c
        ${include_path}/zlib/uncompr.c
        ${include_path}/zlib/trees.c
        ${include_path}/zlib/inftrees.c
        ${include_path}/zlib/inflate.c
        ${include_path}/zlib/inffast.c
        ${include_path}/zlib/infback.c
        ${include_path}/zlib/gzwrite.c
        ${include_path}/zlib/gzread.c
        ${include_path}/zlib/gzlib.c
        ${include_path}/zlib/gzclose.c
        ${include_path}/zlib/deflate.c
        ${include_path}/zlib/crc32.c
        ${include_path}/zlib/compress.c
        ${include_path}/zlib/adler32.c
        ${include_path}/zlib/ioapi.c
        ${include_path}/zlib/unzip.c
        ${include_path}/zlib/dirent.c
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
    )

target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/glew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/half/include
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw3/src
    ${CMAKE_CURRENT_SOURCE_DIR}/nvwa
    ${CMAKE_CURRENT_SOURCE_DIR}/zlib
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/spa
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${platform_specific_include_directories}

    PUBLIC

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

target_link_libraries(${target}
    PRIVATE
    ${PlatformLinkLibs}

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
    )
