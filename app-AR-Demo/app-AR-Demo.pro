##############################################################################
#  File:      app-AR-Demo.pro
#  Purpose:   QMake project definition file for demo application with GLFW
#  Author:    Michael Göttlicher
#  Date:      May 2016
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = app-AR-Demo

CONFIG += console
CONFIG -= qt
CONFIG += glfw
CONFIG += warn_off
CONFIG += app_bundle

DEFINES += "SL_STARTSCENE=C_sceneMeshLoad"

include(../SLProjectCommon.pro)

DESTDIR     = ../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLProject
LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal
LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-assimp
win32 {LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr}

win32 {POST_TARGETDEPS += $$PWD/../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib}
else  {POST_TARGETDEPS += $$PWD/../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a}
   
SOURCES += \
    glfwMain.cpp \
    ARSceneView.cpp \
    ARTracker.cpp \
    ARChessboardTracker.cpp \
    ARArucoTracker.cpp \
    ARCalibration.cpp \
    AR2DMapper.cpp \
    AR2DTracker.cpp
	   
HEADERS += \
    ARSceneView.h \
    ARTracker.h \
    ARChessboardTracker.h \
    ARArucoTracker.h \
    ARCalibration.h \
    AR2DMapper.h \
    AR2DTracker.h

include(../SLProjectCommonLibraries.pro)
include(../SLProjectDeploy.pro)
