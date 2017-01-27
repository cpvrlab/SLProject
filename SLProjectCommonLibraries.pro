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
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_core310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgproc310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_imgcodecs310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_video310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_videoio310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_aruco310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_features2d310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_xfeatures2d310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_calib3d310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_highgui310.lib
    LIBS += $$PWD\_lib\prebuilt\OpenCV\x64\vc12\lib\opencv_flann310.lib

    DEFINES += GLEW_STATIC
    DEFINES += GLEW_NO_GLU
    DEFINES += _GLFW_NO_DLOAD_GDI32
    DEFINES += _GLFW_NO_DLOAD_WINMM
    DEFINES -= UNICODE
    INCLUDEPATH += ../lib-SLExternal/dirent \
}
macx {
    # mac only
    QMAKE_MAC_SDK = macosx10.12
    CONFIG += c++11
    DEFINES += GLEW_NO_GLU
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_core
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_imgproc
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_imgcodecs
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_video
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_videoio
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_aruco
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_features2d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_xfeatures2d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_calib3d
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_highgui
    LIBS += -L$$PWD/_lib/prebuilt/OpenCV/macx -lopencv_flann
    INCLUDEPATH += /usr/include
}
unix:!macx:!android {
    # Install opencv with the following command:
    # sudo apt-get install libopencv-core-dev libopencv-imgproc-dev libopencv-video-dev libopencv-videoio-dev
    OPENCV_LIB_DIRS += /usr/lib #default
    OPENCV_LIB_DIRS += /usr/lib/x86_64-linux-gnu #ubuntu

    CONFIG(release, debug|release) {
        OPENCV_LIB_DIRS += /home/ghm1/libs/opencv-3.1.0/release/lib #ubuntu
    }
    CONFIG(debug, debug|release) {
        OPENCV_LIB_DIRS += /home/ghm1/libs/opencv-3.1.0/debug/lib #ubuntu
    }
    for(dir,OPENCV_LIB_DIRS) {
        !opencv { #If opencv was already found, skip this loop
            CONFIG += opencv
            OPENCV_LIBS =  opencv_core opencv_imgproc opencv_imgproc opencv_video opencv_videoio opencv_calib3d opencv_imgcodecs opencv_aruco opencv_highgui opencv_xfeatures2d opencv_features2d
            #Scan for opencv libs, if one is missing, remove the opencv flag.
            for(lib,OPENCV_LIBS):!exists($$dir/lib$${lib}.so*):CONFIG -= opencv
            opencv {
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

    CONFIG(release, debug|release) {
        INCLUDEPATH += /home/ghm1/libs/opencv-3.1.0/release/include
    }
    CONFIG(debug, debug|release) {
        INCLUDEPATH += /home/ghm1/libs/opencv-3.1.0/debug/include
    }

    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter
}
