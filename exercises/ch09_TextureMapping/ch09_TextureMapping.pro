##############################################################################
#  File:      ch09_TextureMapping.pro
#  Purpose:   QMake project definition file for the Hello Cube demo w. OpenGL
#  Author:    Marcus Hudritsch
#  Date:      September 2012 (HS12)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = ch09_TextureMapping

CONFIG += warn_off
CONFIG -= qml_debug
CONFIG -= qt
CONFIG -= app_bundle

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR     = ../../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM
LIBS       += -L../../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal

include(../../SLProjectCommonLibraries.pro)

INCLUDEPATH += \
    ../include \
    ../../include\
    ../../lib-SLExternal\
    ../../lib-SLExternal/glew/include \
    ../../lib-SLExternal/glfw3/include \
    ../../lib-SLExternal/opencv/include \

HEADERS += \
    ../../include/glUtils.h \
    ../../include/SLCVImage.h \

SOURCES += \
    TextureMapping.cpp \
    ../../lib-SLProject/source/glUtils.cpp \
    ../../lib-SLProject/source/CV/SLCVImage.cpp \
    ../../lib-SLProject/source/SL/SLFileSystem.cpp \
    ../../lib-SLProject/source/SL/SL.cpp

OTHER_FILES += \
../../_data/shaders/ADSTex.frag \
../../_data/shaders/ADSTex.vert \


