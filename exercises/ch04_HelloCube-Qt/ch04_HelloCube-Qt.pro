##############################################################################
#  File:      ch04_HelloCube-Qt.pro
#  Purpose:   QMake project definition file for the Hello Cube demo w. OpenGL
#  Author:    Marcus Hudritsch
#  Date:      September 2016 (HS16)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TEMPLATE = app
TARGET = ch06_HelloCube-Qt

CONFIG += warn_off
CONFIG -= qml_debug
CONFIG -= app_bundle

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM
LIBS += -L../../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal

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
    INCLUDEPATH += ../../lib-SLExternal/png
    INCLUDEPATH += ../../lib-SLExternal/dirent
}
macx {
    # mac only
    QMAKE_MAC_SDK = macosx10.12
    CONFIG += c++11
    DEFINES += GLEW_NO_GLU
    LIBS += -framework OpenGL
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
    INCLUDEPATH += /usr/include
    INCLUDEPATH += ../../lib-SLExternal/png
}
unix:!macx:!android {
    # linux only
    LIBS += -ldl
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
    ../../include\
    ../../lib-SLExternal \
    ../../lib-SLExternal/glew/include \
    ../../lib-SLExternal/glfw3/include \
    ../../lib-SLExternal/half/include \
    ../../lib-SLExternal/zlib \
    ../../lib-SLExternal/jpeg-8 \

SOURCES += \
    main.cpp \
    mainwindow.cpp \

HEADERS += \
    mainwindow.h \

FORMS   += mainwindow.ui
