//#############################################################################
//  File:      qtMain.cpp
//  Purpose:   Implements the main routine with the Qt application instance
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SL.h>
#include "qtGLWidget.h"

#include <QApplication>
#include <QLabel>
#include <QDir>
#include <QFile>
#include <QString>


//-----------------------------------------------------------------------------
void copyPath(QString src, QString dst)
{
    QDir dir(src);
    SL_LOG("1");
    if (!dir.exists())
    {  SL_LOG("src: %s", src.toStdString().c_str());
        return;
    }

    foreach (QString d, dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
    {   QString dst_path = dst + QDir::separator() + d;
        SL_LOG("Try mkpath: %s", dst_path.toStdString().c_str());
        if (dir.mkpath(dst_path))
            SL_LOG("Folder created.");
        copyPath(src+ QDir::separator() + d, dst_path);
    }

    SL_LOG("2");
    foreach (QString f, dir.entryList(QDir::Files))
    {   QString srcFile = src + QDir::separator() + f;
        QString dstFile = dst + QDir::separator() + f;
        SL_LOG("Try: QFile::copy:");
        SL_LOG(srcFile.toStdString().c_str());
        SL_LOG(dstFile.toStdString().c_str());

        bool copied = QFile::copy(srcFile, dstFile);
        if (copied)
            SL_LOG("Success");
    }
    SL_LOG("3");
}
//-----------------------------------------------------------------------------
/*!
The main procedure holding the Qt application instance as well as the main
window instance of our class qt4QMainWindow.
*/
int main(int argc, char *argv[])
{  
    // set command line arguments
    SLVstring cmdLineArgs;
    for(int i = 1; i < argc; i++)
        cmdLineArgs.push_back(argv[i]);

    // main Qt application instance
    QApplication app(argc, argv);

    #if defined(SL_OS_ANDROID)
    //We first need the copy the files from the zipped asset to
    //the application files folder where we can read with fread.
    QDir files("/data/data/ch.fhwn.comgr/files");
    files.mkpath("/data/data/ch.fhwn.comgr/files/shaders");
    files.mkpath("/data/data/ch.fhwn.comgr/files/models");
    files.mkpath("/data/data/ch.fhwn.comgr/files/textures");
    copyPath("assets:" , "/data/data/ch.fhwn.comgr/files");
    #endif

    #ifndef QT_NO_OPENGL
        // on Mac OSX the sample buffers must be turned on for antialiasing
        QGLFormat format;
        format.defaultFormat();
        format.setSampleBuffers(true);
        format.setSwapInterval(1);

        // create OpenGL widget
        qtGLWidget myGLWidget(format, 0, "/data/data/ch.fhwn.comgr/files", cmdLineArgs);
        myGLWidget.resize(640, 480);
        myGLWidget.show();
    #else
        QLabel note("OpenGL Support required");
        note.show();
    #endif
   
    // let's rock ...
    return app.exec();
}
//-----------------------------------------------------------------------------
