##############################################################################
#  File:      app-Viewer-Qt.pro
#  Purpose:   QMake project definition for SLProject Viewer Qt application
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TARGET = app-Viewer-Qt
TEMPLATE = app

CONFIG += console
CONFIG += app_bundle
CONFIG += qt
CONFIG -= glfw

QT += core gui widgets opengl

DEFINES += SL_GUI_QT
DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

include(../SLProjectCommon.pro)

SOURCES += \
   qtMain.cpp \
   qtMainWindow.cpp \
   qtGLWidget.cpp \
   qtPropertyTreeItem.cpp

HEADERS += \
   qtMainWindow.h \
   qtGLWidget.h \
   qtNodeTreeItem.h \
   qtPropertyTreeItem.h \
   qtPropertyTreeWidget.h

FORMS += \
   qtMainWindow.ui

RESOURCES += \
    resources.qrc

