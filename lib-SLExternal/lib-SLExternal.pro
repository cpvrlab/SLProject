##############################################################################
#  File:      lib-SLExternal.pro
#  Purpose:   QMake project definition file for external libraries
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = lib
TARGET = lib-SLExternal

CONFIG += staticlib
CONFIG -= qt
CONFIG += warn_off

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

CONFIG(debug, debug|release) {DEFINES += _DEBUG}

DESTDIR = ../_lib/$$CONFIGURATION/$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

DEFINES += GLEW_NO_GLU

win32 {
    # windows only
    LIBS += -lOpenGL32
    LIBS += -lwinmm
    LIBS += -lgdi32
    LIBS += -luser32
    LIBS += -lkernel32
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_WIN32
    DEFINES += _GLFW_USE_OPENGL
    DEFINES += _GLFW_WGL
    DEFINES += UNICODE
    DEFINES += _UNICODE
}
macx {
    # mac only
    QMAKE_MAC_SDK = macosx10.13
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_COCOA
    DEFINES += _GLFW_NSGL
    DEFINES += _GLFW_USE_OPENGL

    LIBS += -framework OpenGL
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework QuartzCore

    CONFIG -= app_bundle
    CONFIG += C/ObjC
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}
unix:!macx:!android {
    # linux only
    DEFINES += _GLFW_X11
    DEFINES += _GLFW_GLX
    DEFINES += _GLFW_USE_OPENGL
    DEFINES += _GLFW_HAS_GLXGETPROCADDRESS
    LIBS += -lgl
    LIBS += -lX11
    LIBS += -lXxf86vm
    LIBS += -lXcursor
    QMAKE_CXXFLAGS += -ldl
    QMAKE_CXXFLAGS += -lpthread
    QMAKE_CXXFLAGS += -Wunused-parameter
}
android {
    # android only
    QMAKE_CXXFLAGS += -Wunused-parameter
}


INCLUDEPATH += \
    ../include \
    glew/include \
    glfw3/include \
    half/include \
    glfw3/src \
    nvwa \
    randomc \
    zlib \
    imgui \
    spa \

HEADERS += \
    glew/include/GL/glew.h \
    glfw3/include/GLFW/glfw3.h \
    glfw3/include/GLFW/glfw3native.h \
    glfw3/src/glfw_config.h \
    glfw3/src/internal.h \
    randomc/randomc.h \
    randomc/random.h \
    Shoemake/Decompose.h \
    Shoemake/EulerAngles.h \
    Shoemake/TypeDefs.h \
    imgui/imconfig.h \
    imgui/imgui_internal.h \
    imgui/imgui.h \
    imgui/stb_rect_pack.h \
    imgui/stb_textedit.h \
    imgui/stb_truetype.h \
    spa/spa.h

SOURCES += \
    glew/src/glew.c \
    glfw3/src/context.c \
    glfw3/src/init.c \
    glfw3/src/input.c \
    glfw3/src/monitor.c \
    glfw3/src/window.c \
    randomc/sobol.cpp \
    randomc/ranrotw.cpp \
    randomc/ranrotb.cpp \
    randomc/random.cpp \
    randomc/mother.cpp \
    randomc/mersenne.cpp \
    Shoemake/EulerAngles.cpp \
    Shoemake/Decompose.cpp \
    imgui/imgui_draw.cpp \
    imgui/imgui.cpp \
    spa/spa.cpp

macx {
#Mac OSX only --------------------------------------
HEADERS += \
    glfw3/src/cocoa_platform.h \
    glfw3/src/iokit_joystick.h \
    glfw3/src/posix_tls.h \
    glfw3/src/nsgl_context.h \
    zlib/ioapi.h \
    zlib/unzip.h \
    zlib/zconf.in.h \
    zlib/zutil.h \
    zlib/zlib.h \
    zlib/zconf.h \
    zlib/trees.h \
    zlib/inftrees.h \
    zlib/inflate.h \
    zlib/inffixed.h \
    zlib/inffast.h \
    zlib/gzguts.h \
    zlib/deflate.h \
    zlib/crypt.h \
    zlib/crc32.h \

SOURCES += \
    glfw3/src/mach_time.c \
    glfw3/src/posix_tls.c \
    zlib/zutil.c \
    zlib/uncompr.c \
    zlib/trees.c \
    zlib/inftrees.c \
    zlib/inflate.c \
    zlib/inffast.c \
    zlib/infback.c \
    zlib/gzwrite.c \
    zlib/gzread.c \
    zlib/gzlib.c \
    zlib/gzclose.c \
    zlib/deflate.c \
    zlib/crc32.c \
    zlib/compress.c \
    zlib/adler32.c \
    zlib/ioapi.c \
    zlib/unzip.c \

OBJECTIVE_SOURCES += \
    glfw3/src/cocoa_init.m \
    glfw3/src/cocoa_monitor.m \
    glfw3/src/cocoa_window.m \
    glfw3/src/iokit_joystick.m \
    glfw3/src/nsgl_context.m \

}
win32 {
#Windows only -------------------------------------
INCLUDEPATH += dirent

HEADERS += \
    glfw3/src/win32_platform.h \
    glfw3/src/win32_tls.h \
    glfw3/src/winmm_joystick.h \
    glfw3/src/wgl_context.h \
    nvwa/debug_new.h \
    zlib/ioapi.h \
    zlib/unzip.h \
    zlib/zconf.in.h \
    zlib/zutil.h \
    zlib/zlib.h \
    zlib/zconf.h \
    zlib/trees.h \
    zlib/inftrees.h \
    zlib/inflate.h \
    zlib/inffixed.h \
    zlib/inffast.h \
    zlib/gzguts.h \
    zlib/deflate.h \
    zlib/crypt.h \
    zlib/crc32.h \
    dirent/dirent.h \

SOURCES += \
    glfw3/src/win32_init.c \
    glfw3/src/win32_monitor.c \
    glfw3/src/win32_time.c \
    glfw3/src/win32_tls.c \
    glfw3/src/win32_window.c \
    glfw3/src/winmm_joystick.c \
    glfw3/src/wgl_context.c \
    nvwa/debug_new.cpp \
    zlib/zutil.c \
    zlib/uncompr.c \
    zlib/trees.c \
    zlib/inftrees.c \
    zlib/inflate.c \
    zlib/inffast.c \
    zlib/infback.c \
    zlib/gzwrite.c \
    zlib/gzread.c \
    zlib/gzlib.c \
    zlib/gzclose.c \
    zlib/deflate.c \
    zlib/crc32.c \
    zlib/compress.c \
    zlib/adler32.c \
    zlib/ioapi.c \
    zlib/unzip.c \
    dirent/dirent.c \

}
unix:!macx:!android {
#Linux only -------------------------
INCLUDEPATH += \
    glfw3/include \
    glfw3/src

HEADERS += \
    glfw3/src/glfw_config.h \
    glfw3/src/x11_platform.h \
    glfw3/src/xkb_unicode.h \
    glfw3/src/linux_joystick.h \
    glfw3/src/posix_time.h \
    glfw3/src/posix_tls.h \
    glfw3/src/glx_context.h \

SOURCES += \
    glfw3/src/x11_init.c \
    glfw3/src/x11_monitor.c \
    glfw3/src/x11_window.c \
    glfw3/src/xkb_unicode.c \
    glfw3/src/linux_joystick.c \
    glfw3/src/posix_time.c \
    glfw3/src/posix_tls.c \
    glfw3/src/glx_context.c \

}


