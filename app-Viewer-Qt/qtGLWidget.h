//#############################################################################
//  File:      qtGLWidget.h
//  Purpose:   Declaration of the QGLWidget derive window class.
//             In general all the event hadlers will foreward the events to the 
//             appropriate event handlers of the SLSceneView class
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTGLWIDGET_H
#define QTGLWIDGET_H

#include <QGLWidget>

class qtMainWindow;

//-----------------------------------------------------------------------------
//! Qt QGLWidget class (for OpenGL rendering) for OpenGL display
/*!
qtGLWidget provides the GUI functionality for OpenGL rendering and is
therefore derived from QGLWidget. It forewards all rendering and interaction
tasks to the according methods of the interface in slInterface
*/
class qtGLWidget : public QGLWidget
{
    public:                 //! Constructor for first GLWidget for the main window
                            qtGLWidget           (QGLFormat &format,
                                                  QWidget* parent,
                                                  SLstring appPath,
                                                  SLVstring cmdLineArgs);

                            //! Contructor for subsequent GLWidgets
                            qtGLWidget           (QWidget* parent,
                                                  QGLWidget* shareWidget);

                           ~qtGLWidget           ();

            void            initializeGL         ();
            void            resizeGL             (int w,int h);
            void            paintGL              ();
            void            mousePressEvent      (QMouseEvent *e);
            void            mouseReleaseEvent    (QMouseEvent *e);
            void            mouseDoubleClickEvent(QMouseEvent *e);
            void            mouseMoveEvent       (QMouseEvent *e);
            void            wheelEvent           (QWheelEvent *e);
            void            keyPressEvent        (QKeyEvent   *e);
            void            keyReleaseEvent      (QKeyEvent   *e);

            qtGLWidget*     splitActive          (Qt::Orientation orientation);

            // Getters
            SLSceneView*    sv                   () {return _sv;}
            int             svIndex              () {return _svIndex;}

    static  qtMainWindow*   mainWindow;    //!< pointer to the main window

    private:
            SLSceneView*    _sv;           //!< pointer to the corresponding sceneview
            int             _svIndex;      //!< index of element in the SLScene::sceneviews vector
            SLstring        _appPath;      //!< application path
            SLVstring       _cmdLineArgs;  //!< command line arguments
            SLVec2i         _touch2;       //!< Last finger touch 2 position in pixels
            SLVec2i         _touchDelta;   //!< Delta between two fingers in x
};
//-----------------------------------------------------------------------------
#endif //QTGLWIDGET_H
