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
CONFIG += app_bundle

DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

include(../SLProjectCommon.pro)
   
SOURCES += \
   glfwMain.cpp \
   NewNodeSceneView.cpp
	   
HEADERS += \
    NewNodeSceneView.h
