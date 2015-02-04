##############################################################################
#  File:      app-LeapMotionTest.pro
#  Purpose:   QMake project definition file for demo application with GLFW
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = app-LeapMotionTest

CONFIG += console
CONFIG -= qt
CONFIG += glfw
CONFIG += warn_off

DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}
#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLProject
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-assimp
win32:LIBS += ws2_32.lib
win32:LIBS += Setupapi.lib
win32:LIBS += Shell32.lib
win32:LIBS += ../lib-SLExternal/leapmotion/lib/$$PLATFORM/Leap.lib
#LIBS += ../lib-SLExternal/leapmotion/lib/$$PLATFORM/libLeap.dylib
macx:LIBS += ../_bin-$$CONFIGURATION-$$PLATFORM/libLeap.dylib
macx|win32 {LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr}

win32 {POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib}
else  {POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a}

include(../SLProjectCommon.pro)

INCLUDEPATH += \
    ../lib-SLExternal/leapmotion/include
   
SOURCES += \
    glfwMain.cpp \
    CustomSceneview.cpp \
    SLLeapController.cpp \
    SLLeapFinger.cpp \
    SLLeapGesture.cpp \
    SLLeapHand.cpp \
    SLLeapTool.cpp
	   

HEADERS += \
    CustomSceneView.h \
    SampleListeners.h \
    SLLeapController.h \
    SLLeapFinger.h \
    SLLeapGesture.h \
    SLLeapHand.h \
    SLLeapTool.h
