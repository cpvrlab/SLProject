##############################################################################
#  File:      SLProject_common.pro
#  Purpose:   QMake project definition for common SLProject projects
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

CONFIG += warn_off
CONFIG -= qml_debug

CONFIG(qt) {
   QT += core gui widgets opengl
   DEFINES += SL_GUI_QT
}

CONFIG(glfw) {
   CONFIG -= qt
   DEFINES += SL_GUI_GLFW
}


CONFIG(debug, debug|release) {DEFINES += _DEBUG}

win32 {
    # windows only
    LIBS += -lOpenGL32
    LIBS += -lwinmm
    LIBS += -lgdi32
    LIBS += -luser32
    LIBS += -lkernel32
    LIBS += -lshell32
    LIBS += -lsetupapi
    LIBS += -lws2_32
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    INCLUDEPATH += ../lib-SLExternal/png
}
macx {
    # mac only
    CONFIG += c++11
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
    INCLUDEPATH += ../lib-SLExternal/png
}
unix:!macx:!android {
    # linux only
    LIBS += -lGL
    LIBS += -lX11
    LIBS += -lXrandr    #livrandr-dev
    LIBS += -lXi        #libxi-dev
    LIBS += -lXinerama  #libxinerama-dev
    LIBS += -lXxf86vm   #libxf86vm
    LIBS += -lXcursor
    LIBS += -ludev      #libudev-dev
    LIBS += -lpthread   #libpthread
    LIBS += -lpng
    LIBS += -lz
    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}
android {
    # Android only (not yet used)
    # The Android project can not be built so far with Qt5
    # The Android project is built with VS-Android in VisualStudio
    ANDROID_PACKAGE_SOURCE_DIR = $$PWD/GUI/AndroidQt
    message($$ANDROID_PACKAGE_SOURCE_DIR)
}


INCLUDEPATH += \
    ../include\
    ../lib-SLExternal\
    ../lib-SLExternal/assimp/include \
    ../lib-SLExternal/assimp/code \
    ../lib-SLExternal/glew/include \
    ../lib-SLExternal/glfw3/include \
    ../lib-SLExternal/zlib\
    ../lib-SLExternal/randomc \
    ../lib-SLExternal/nvwa \
    ../lib-SLExternal/jpeg-8\
    ../lib-SLExternal/oculus/LibOVR/Include \
    include


# OpenCV
HASOPENCV = No
win32 {
    # windows only
    exists($(OPENCV_DIR)/lib/opencv_*.lib) {
        DEFINES += HAS_OPENCV
        INCLUDEPATH += c:\Lib\opencv\build\install\include
        LIBS += $(OPENCV_DIR)\lib\opencv_core300d.lib
        LIBS += $(OPENCV_DIR)\lib\opencv_imgproc300d.lib
        LIBS += $(OPENCV_DIR)\lib\opencv_video300d.lib
        LIBS += $(OPENCV_DIR)\lib\opencv_videoio300d.lib
        HASOPENCV = Yes
    }
}
macx {
    # mac only
    exists(/usr/local/lib/libopencv_*.dylib) {
        DEFINES += HAS_OPENCV
        INCLUDEPATH += /usr/local/include/
        LIBS += -L/usr/local/lib -lopencv_core
        LIBS += -L/usr/local/lib -lopencv_imgproc
        LIBS += -L/usr/local/lib -lopencv_video
        LIBS += -L/usr/local/lib -lopencv_videoio
        HASOPENCV = Yes
    }
}
unix:!macx:!android {
    # linux only: Install opencv with the following command:
    # sudo apt-get install libopencv-core-dev libopencv-imgproc-dev libopencv-video-dev libopencv-videoio-dev

    OPENCV_LIB_DIRS += /usr/lib #default
    OPENCV_LIB_DIRS += /usr/lib/x86_64-linux-gnu #ubuntu

    for(dir,OPENCV_LIB_DIRS) {
        exists($$dir/libopencv_*.so) {
            CONFIG += opencv
            DEFINES += HAS_OPENCV
            INCLUDEPATH += /usr/include/
            LIBS += -L$$dir -lopencv_core -lopencv_imgproc -lopencv_imgproc -lopencv_video -lopencv_videoio
            HASOPENCV = Yes
        }
    }
}

message(-----------------------------------------)
message(Target: $$TARGET)
message(Config: $$CONFIGURATION)
message(Platform: $$PLATFORM)
message(Has OpenCV: $$HASOPENCV)
