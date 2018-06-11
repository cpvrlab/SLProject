##############################################################################
#  File:      cv08_MeshMorphing
#  Purpose:   QMake project definition file for mesh morphing in OpenCV
#  Author:    Marcus Hudritsch
#  Date:      Spring 2017 (HS17)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = cv08_MeshMorphing

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

include(../../SLProjectCommonLibraries.pro)

INCLUDEPATH += ../../lib-SLExternal/opencv/include

SOURCES += cv08_MeshMorphing.cpp
