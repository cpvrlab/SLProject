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

DEFINES += SL_GUI_QT
DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

include(../SLProjectCommon.pro)

DESTDIR     = ../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLProject
LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal
LIBS += -L$$PWD/../_lib/$$CONFIGURATION/$$PLATFORM -llib-assimp
macx|win32 {LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr}

win32 {POST_TARGETDEPS += $$PWD/../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib}
else  {POST_TARGETDEPS += $$PWD/../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a}

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

include(../SLProjectCommonLibraries.pro)
include(../SLProjectDeploy.pro)

