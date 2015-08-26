##############################################################################
#  File:      SLProjectCommon.pro
#  Purpose:   QMake project definition for common SLProject projects
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

CONFIG += warn_off
CONFIG -= qml_debug
CONFIG(debug, debug|release) {DEFINES += _DEBUG}

CONFIG(qt) {
   QT += core gui widgets opengl
   DEFINES += SL_GUI_QT
}

CONFIG(glfw) {
   CONFIG -= qt
   DEFINES += SL_GUI_GLFW
}

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

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
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_core300.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgproc300.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_video300.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_videoio300.lib
    DEFINES += GLEW_STATIC
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    DEFINES += SL_HAS_OPENCV
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
    LIBS += -L../_lib/prebuilt/OpenCV/macx -lopencv_core
    LIBS += -L../_lib/prebuilt/OpenCV/macx -lopencv_imgproc
    LIBS += -L../_lib/prebuilt/OpenCV/macx -lopencv_video
    LIBS += -L../_lib/prebuilt/OpenCV/macx -lopencv_videoio
    INCLUDEPATH += ../lib-SLExternal/png
    DEFINES += SL_HAS_OPENCV
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

    # Install opencv with the following command:
    # sudo apt-get install libopencv-core-dev libopencv-imgproc-dev libopencv-video-dev libopencv-videoio-dev
    OPENCV_LIB_DIRS += /usr/lib #default
    OPENCV_LIB_DIRS += /usr/lib/x86_64-linux-gnu #ubuntu

    for(dir,OPENCV_LIB_DIRS) {
        exists($$dir/libopencv_*.so) {
            CONFIG += opencv
            DEFINES += SL_HAS_OPENCV
            INCLUDEPATH += /usr/include/
            LIBS += -L$$dir -lopencv_core -lopencv_imgproc -lopencv_imgproc -lopencv_video -lopencv_videoio
        }
    }
}

INCLUDEPATH += \
    include \
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
    ../lib-SLExternal/opencv/include \

