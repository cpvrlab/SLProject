##############################################################################
#  File:      SLProjectCommonLibraries.pro
#  Purpose:   QMake project definition for common SLProject projects
#  Author:    Marcus Hudritsch, Manuel Frischknecht
#  Date:      August 2015
#  Copyright: Marcus Hudritsch, Manuel Frischknecht, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

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
    DEFINES += GLEW_NO_GLU
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    DEFINES += SL_HAS_OPENCV
    INCLUDEPATH += ../lib-SLExternal/png \
                   ../lib-SLExternal/dirent \

}
macx {
    # mac only
    QMAKE_MAC_SDK = macosx10.11
    CONFIG += c++11
    DEFINES += GLEW_NO_GLU
   #LIBS += -framework Foundation
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
    INCLUDEPATH += /usr/include
    DEFINES += SL_HAS_OPENCV
}
unix:!macx:!android {
    # Install opencv with the following command:
    # sudo apt-get install libopencv-core-dev libopencv-imgproc-dev libopencv-video-dev libopencv-videoio-dev
    OPENCV_LIB_DIRS += /usr/lib #default
    OPENCV_LIB_DIRS += /usr/lib/x86_64-linux-gnu #ubuntu
    for(dir,OPENCV_LIB_DIRS) {
        !opencv { #If opencv was already found, skip this loop
            CONFIG += opencv
            OPENCV_LIBS =  opencv_core opencv_imgproc opencv_imgproc opencv_video opencv_videoio
            #Scan for opencv libs, if one is missing, remove the opencv flag.
            for(lib,OPENCV_LIBS):!exists($$dir/lib$${lib}.so*):CONFIG -= opencv
            opencv {
                DEFINES += SL_HAS_OPENCV
                INCLUDEPATH += /usr/include/
                LIBS += -L$$dir
                for(lib,OPENCV_LIBS) LIBS += -l$$lib
            }
            unset(OPENCV_LIBS)
        }
    }
    !opencv:warning(OpenCV is either not installed or not up to date (install OpenCV 3.0))

    # linux only
    LIBS += -ldl
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
