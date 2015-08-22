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

LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLProject
LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal
LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-assimp
macx|win32 {LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr}

win32 {POST_TARGETDEPS += $$PWD/../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib}
else  {POST_TARGETDEPS += $$PWD/../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a}

INCLUDEPATH += \
    include

HEADERS += \
   include/qtGLWidget.h

SOURCES += \
   source/qtGLWidget.cpp \
   source/qtMain.cpp
