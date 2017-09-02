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

include(../SLProjectCommon.pro)
include(../SLProjectCommonLibraries.pro)

SOURCES += VolumeRendering.cpp
SOURCES += ../lib-SLProject/source/glUtils.cpp
SOURCES += ../lib-SLProject/source/CV/SLCVImage.cpp
SOURCES += ../lib-SLProject/source/SL/SL.cpp
SOURCES += ../lib-SLProject/source/SL/SLFileSystem.cpp

HEADERS += ../include/glUtils.h
HEADERS += ../include/SLCV.h
HEADERS += ../include/SLCVImage.h

OTHER_FILES += VolumeRenderingRayCast.vert
OTHER_FILES += VolumeRenderingSiddon_TF.frag
OTHER_FILES += VolumeRenderingSiddon_MIP.frag
OTHER_FILES += VolumeRenderingSampling_TF.frag
OTHER_FILES += VolumeRenderingSampling_MIP.frag
OTHER_FILES += VolumeRenderingSlicing.vert
OTHER_FILES += VolumeRenderingSlicing.frag
