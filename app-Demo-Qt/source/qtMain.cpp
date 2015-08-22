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
/*!
The main procedure holding the Qt application instance as well as the main
window instance of our class qt4QMainWindow.
*/
int main(int argc, char *argv[])
{  
    // set command line arguments
    SLVstring cmdLineArgs;
    for(int i = 0; i < argc; i++)
        cmdLineArgs.push_back(argv[i]);

    // main Qt application instance
    QApplication app(argc, argv);

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
