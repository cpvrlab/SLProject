##############################################################################
#  File:      app-NodeManipulationDemo.pro
#  Purpose:   QMake project definition file for demo application with GLFW
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = app-NewNodeTesting

CONFIG += console
CONFIG -= qt
CONFIG += glfw
CONFIG += warn_off
CONFIG -= app_bundle

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

macx|win32 {LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr}

win32 {POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib}
else  {POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a}

include(../SLProjectCommon.pro)
   
SOURCES += \
   glfwMain.cpp \
   NewNodeSceneView.cpp
	   

HEADERS += \
    NewNodeSceneView.h
