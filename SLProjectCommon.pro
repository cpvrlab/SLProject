##############################################################################
#  File:      SLProjectCommon.pro
#  Purpose:   QMake project definition for common SLProject projects
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

CONFIG += warn_off
CONFIG -= qml_debug
CONFIG(debug, debug|release) {DEFINES += _DEBUG}

DEFINES += SL_USES_CVCAPTURE

CONFIG(qt) {
   QT += core gui widgets opengl
   DEFINES += SL_GUI_QT
}

CONFIG(glfw) {
   CONFIG -= qt
   DEFINES += SL_GUI_GLFW
}

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

CONFIG += c++11
unix:!macx:!android:QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += \
    include \
    ../include\
    ../lib-SLExternal\
    ../lib-SLExternal/assimp/include \
    ../lib-SLExternal/assimp/code \
    ../lib-SLExternal/glew/include \
    ../lib-SLExternal/glfw3/include \
    ../lib-SLExternal/randomc \
    ../lib-SLExternal/nvwa \
    ../lib-SLExternal/oculus/LibOVR/Include \
    ../lib-SLExternal/opencv/include \
    ../lib-SLExternal/imgui \
    ../lib-SLExternal/spa \

