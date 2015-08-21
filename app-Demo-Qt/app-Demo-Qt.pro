##############################################################################
#  File:      app-Demo-Qt.pro
#  Purpose:   QMake project definition file for demo application with Qt
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = app-Demo-Qt

CONFIG += console
CONFIG += qt
CONFIG -= glfw
CONFIG += app_bundle

QT += core gui widgets opengl

DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

include(../SLProjectCommon.pro)

INCLUDEPATH += \
    include

HEADERS += \
   include/qtGLWidget.h

SOURCES += \
   source/qtGLWidget.cpp \
   source/qtMain.cpp
