##############################################################################
#  File:      app-Demo-GLFW.pro
#  Purpose:   QMake project definition file for demo application with GLFW
#  Author:    Marcus Hudritsch
#  Date:      August 2015
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = app
TARGET = app-Demo-GLFW-OPENCV

CONFIG += console
CONFIG -= qt
CONFIG += glfw
CONFIG += warn_off
CONFIG -= app_bundle

DEFINES += "SL_STARTSCENE=cmdSceneRevolver"

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


message(Configuration: $$CONFIGURATION)
message(Plattform: $$PLATFORM)

#OpenCV

win32 {
    # windows only
    exists($(OPENCV_DIR)/lib/opencv_*.lib) {
        message("OpenCV: Yes")
        DEFINES += HAS_OPENCV
        INCLUDEPATH += c:\Lib\opencv\build\install\include
        LIBS += $(OPENCV_DIR)\lib\opencv_core300d.lib
        LIBS += $(OPENCV_DIR)\lib\opencv_imgproc300d.lib
        LIBS += $(OPENCV_DIR)\lib\opencv_video300d.lib
        LIBS += $(OPENCV_DIR)\lib\opencv_videoio300d.lib
    } else {message("OpenCV: No")}
}
macx {
    # mac only
    exists(/usr/local/lib/libopencv_*.dylib) {
        message("OpenCV: Yes")
        DEFINES += HAS_OPENCV
        INCLUDEPATH += /usr/local/include/
        LIBS += -L/usr/local/lib -lopencv_core
        LIBS += -L/usr/local/lib -lopencv_imgproc
        LIBS += -L/usr/local/lib -lopencv_video
        LIBS += -L/usr/local/lib -lopencv_videoio
    } else {message("OpenCV: No")}
}
unix:!macx:!android {
    # linux only: Install opencv with the following command:
    # sudo apt-get install libopencv-core-dev libopencv-imgproc-dev libopencv-video-dev libopencv-videoio-dev
    message("OpenCV: No")
}

include(../SLProjectCommon.pro)

SOURCES += \
    source/glfwMain-OPENCV.cpp
	   
