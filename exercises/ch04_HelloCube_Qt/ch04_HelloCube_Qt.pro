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
TARGET = ch06_HelloCube_Qt

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

include(../../SLProjectCommonLibraries.pro)

INCLUDEPATH += \
    ../../include\
    ../../lib-SLExternal \
    ../../lib-SLExternal/glew/include \

SOURCES += HelloCube_Qt.cpp

HEADERS += HelloCube_Qt.h

FORMS += HelloCube_Qt.ui
