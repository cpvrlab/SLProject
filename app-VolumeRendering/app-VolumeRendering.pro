##############################################################################
#  File:      app-VolumeRendering.pro
#  Purpose:   QMake project definition file for the Hello Cube demo w. OpenGL
#  Author:    Marcus Hudritsch
#  Date:      September 2012 (HS12)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = VolumeRendering

CONFIG += warn_off
CONFIG -= qml_debug
CONFIG -= qt
CONFIG -= app_bundle
CONFIG += glfw

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal

win32 {
    # windows only
    LIBS += -lOpenGL32
    LIBS += -lwinmm
    LIBS += -lgdi32
    LIBS += -luser32
    LIBS += -lkernel32
    LIBS += -lshell32
    LIBS += -lsetupapi
    LIBS += -lws2_32
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    INCLUDEPATH += ../lib-SLExternal/png
}
macx {
    # mac only
    CONFIG += c++11
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
    INCLUDEPATH += ../lib-SLExternal/png
}
unix:!macx:!android {
    # linux only
    LIBS += -lGL
    LIBS += -lX11
    LIBS += -lXrandr    #livrandr-dev
    LIBS += -lXi        #libxi-dev
    LIBS += -lXinerama  #libxinerama-dev
    LIBS += -lXxf86vm   #libxf86vm
    LIBS += -lXcursor
    LIBS += -ludev      #libudev-dev
    LIBS += -lpthread   #libpthread
    LIBS += -lpng
    LIBS += -lz
    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}

INCLUDEPATH += \
    ../include\
    ../lib-SLExternal \
    ../lib-SLExternal/dirent \
    ../lib-SLExternal/glew/include \
    ../lib-SLExternal/glfw3/include \
    ../lib-SLExternal/zlib \
    ../lib-SLExternal/jpeg-8 \

SOURCES += VolumeRendering.cpp
SOURCES += glUtils.cpp 
SOURCES += ../lib-SLProject/source/SL/SLImage.cpp 
SOURCES += ../lib-SLProject/source/SL/SL.cpp

HEADERS += glUtils.h
HEADERS += ../include/SLImage.h

OTHER_FILES += VolumeRenderingRayCast.vert
OTHER_FILES += VolumeRenderingSiddon_TF.frag
OTHER_FILES += VolumeRenderingSiddon_MIP.frag
OTHER_FILES += VolumeRenderingSampling_TF.frag
OTHER_FILES += VolumeRenderingSampling_MIP.frag
OTHER_FILES += VolumeRenderingSlicing.vert
OTHER_FILES += VolumeRenderingSlicing.frag
