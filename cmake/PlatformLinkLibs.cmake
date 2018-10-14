#
# CMake configuration for platform specific link libraries for all projects
#

set(PlatformLinkLibs)

#==============================================================================
if("${SYSTEM_NAME_UPPER}" STREQUAL "LINUX")
    set(PlatformLinkLibs)

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "WINDOWS") #----------------------------
    set(PlatformLinkLibs
        OpenGL32
        winmm
        gdi32
        user32
        kernel32
        shell32
        setupapi
        ws2_32)

elseif("${SYSTEM_NAME_UPPER}" STREQUAL "DARWIN") #-----------------------------
    FIND_LIBRARY(COCOA_LIB Cocoa)
    FIND_LIBRARY(IOKIT_LIB IOKit)
    FIND_LIBRARY(OPENGL_LIB OpenGL)
    FIND_LIBRARY(QUARZ_LIB QuartzCore)

    set(PlatformLinkLibs
        ${COCOA_LIB}
        ${IOKIT_LIB}
        ${OPENGL_LIB}
        ${QUARZ_LIB})

endif()
#==============================================================================
