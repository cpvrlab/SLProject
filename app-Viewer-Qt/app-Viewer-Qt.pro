#-------------------------------------------------
#
# Project created by QtCreator 2014-06-27T07:48:09
#
#-------------------------------------------------

TARGET = app-Viewer-Qt
TEMPLATE = app

CONFIG += console
CONFIG += warn_off
CONFIG -= qml_debug

QT += core gui widgets opengl

DEFINES += SL_GUI_QT
DEFINES += "SL_STARTSCENE=cmdSceneMeshLoad"

include(../SLProjectCommon.pro)

DESTDIR = ../_bin-$$CONFIGURATION-$$PLATFORM
OBJECTS_DIR = ../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-assimp
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLExternal
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-SLProject
LIBS += -L../_lib/$$CONFIGURATION/$$PLATFORM -llib-ovr
win32 {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/lib-SLProject.lib
} else {
  POST_TARGETDEPS += ../_lib/$$CONFIGURATION/$$PLATFORM/liblib-SLProject.a
}

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
