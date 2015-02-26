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
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_COCOA
    DEFINES += _GLFW_NSGL
    DEFINES += _GLFW_USE_OPENGL

    CONFIG -= app_bundle
    CONFIG += C/ObjC
    QMAKE_CXXFLAGS += -Wno-unused-parameter

    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
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
    QMAKE_CXXFLAGS += -Wunused-parameter
}
android {
    # android only
    QMAKE_CXXFLAGS += -Wunused-parameter
}


INCLUDEPATH += \
    glew/include \
    glfw3/include \
    glfw3/src \
    png \
    jpeg-8 \
    nvwa \
    randomc \
    zlib \

HEADERS += \
    glew/include/GL/glew.h \
    glfw3/include/GLFW/glfw3.h \
    glfw3/include/GLFW/glfw3native.h \
    glfw3/src/glfw_config.h \
    glfw3/src/internal.h \
    jpeg-8/jversion.h \
    jpeg-8/jpeglib.h \
    jpeg-8/jpegint.h \
    jpeg-8/jmorecfg.h \
    jpeg-8/jmemsys.h \
    jpeg-8/jinclude.h \
    jpeg-8/jerror.h \
    jpeg-8/jdct.h \
    jpeg-8/jconfig.h \
    randomc/randomc.h \
    randomc/random.h \
    Shoemake/Decompose.h \
    Shoemake/EulerAngles.h \
    Shoemake/TypeDefs.h \

SOURCES += \
    glew/src/glew.c \
    glfw3/src/context.c \
    glfw3/src/init.c \
    glfw3/src/input.c \
    glfw3/src/monitor.c \
    glfw3/src/window.c \
    jpeg-8/jutils.c \
    jpeg-8/jquant2.c \
    jpeg-8/jquant1.c \
    jpeg-8/jmemnobs.c \
    jpeg-8/jmemmgr.c \
    jpeg-8/jidctint.c \
    jpeg-8/jidctfst.c \
    jpeg-8/jidctflt.c \
    jpeg-8/jfdctint.c \
    jpeg-8/jfdctfst.c \
    jpeg-8/jfdctflt.c \
    jpeg-8/jerror.c \
    jpeg-8/jdtrans.c \
    jpeg-8/jdsample.c \
    jpeg-8/jdpostct.c \
    jpeg-8/jdmerge.c \
    jpeg-8/jdmaster.c \
    jpeg-8/jdmarker.c \
    jpeg-8/jdmainct.c \
    jpeg-8/jdinput.c \
    jpeg-8/jdhuff.c \
    jpeg-8/jddctmgr.c \
    jpeg-8/jdcolor.c \
    jpeg-8/jdcoefct.c \
    jpeg-8/jdatasrc.c \
    jpeg-8/jdatadst.c \
    jpeg-8/jdarith.c \
    jpeg-8/jdapistd.c \
    jpeg-8/jdapimin.c \
    jpeg-8/jctrans.c \
    jpeg-8/jcsample.c \
    jpeg-8/jcprepct.c \
    jpeg-8/jcparam.c \
    jpeg-8/jcomapi.c \
    jpeg-8/jcmaster.c \
    jpeg-8/jcmarker.c \
    jpeg-8/jcmainct.c \
    jpeg-8/jcinit.c \
    jpeg-8/jchuff.c \
    jpeg-8/jcdctmgr.c \
    jpeg-8/jccolor.c \
    jpeg-8/jccoefct.c \
    jpeg-8/jcarith.c \
    jpeg-8/jcapistd.c \
    jpeg-8/jcapimin.c \
    jpeg-8/jaricom.c \
    randomc/sobol.cpp \
    randomc/ranrotw.cpp \
    randomc/ranrotb.cpp \
    randomc/random.cpp \
    randomc/mother.cpp \
    randomc/mersenne.cpp \
    Shoemake/EulerAngles.cpp \
    Shoemake/Decompose.cpp \

win32 { #Windows only -------------------------------------

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
    png/pngwutil.c \
    png/pngwtran.c \
    png/pngwrite.c \
    png/pngwio.c \
    png/pngtrans.c \
    png/pngset.c \
    png/pngrutil.c \
    png/pngrtran.c \
    png/pngrio.c \
    png/pngread.c \
    png/pngpread.c \
    png/pngmem.c \
    png/pngget.c \
    png/pngerror.c \
    png/png.c \
}
unix:!macx:!android { #Linux only -------------------------
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
macx { #Mac OSX only --------------------------------------
HEADERS += \
    glfw3/src/cocoa_platform.h \
    glfw3/src/iokit_joystick.h \
    glfw3/src/posix_tls.h \
    glfw3/src/nsgl_context.h \
    #nvwa/debug_new.h \
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
    #nvwa/debug_new.cpp \
    png/pngwutil.c \
    png/pngwtran.c \
    png/pngwrite.c \
    png/pngwio.c \
    png/pngtrans.c \
    png/pngset.c \
    png/pngrutil.c \
    png/pngrtran.c \
    png/pngrio.c \
    png/pngread.c \
    png/pngpread.c \
    png/pngmem.c \
    png/pngget.c \
    png/pngerror.c \
    png/png.c \
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
