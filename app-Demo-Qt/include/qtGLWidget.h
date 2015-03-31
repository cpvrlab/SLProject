//#############################################################################
//  File:      qtGLWidget.h
//  Purpose:   Declaration of the QGLWidget derive window class.
//             In general all the event handlers will forward the events to the 
//             appropriate event handlers of the SLSceneView class
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTGLWIDGET_H
#define QTGLWIDGET_H

#include <SL.h>
#include <SLVec2.h>
#include <QGLWidget>

class SLSceneView;

//-----------------------------------------------------------------------------
//! Qt QGLWidget class (for OpenGL rendering) for OpenGL display
/*!
qtGLWidget provides the GUI functionality for OpenGL rendering and is
therefore derived from QGLWidget. It forwards all rendering and interaction
tasks to the according methods of the interface in slInterface
*/
class qtGLWidget : public QGLWidget
{
    Q_OBJECT
    public:                 //! Constructor for first GLWidget for the main window
                            qtGLWidget           (QGLFormat &format,
                                                  QWidget* parent,
                                                  SLstring appPath,
                                                  SLVstring cmdLineArgs);

                            //! Constructor for subsequent GLWidgets
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

            SLSceneView*    sv                   () {return _sv;}
            int             svIndex              () {return _svIndex;}

    private slots:
            void            longTouch            ();

    private:
            SLSceneView*    _sv;           //!< pointer to the corresponding sceneview
            int             _svIndex;      //!< index of element in the SLScene::sceneviews vector
            SLstring        _appPath;      //!< application path
            SLVstring       _cmdLineArgs;  //!< command line arguments
            SLVec2i         _touchStart;   //!< touch start position in pixels
            SLVec2i         _touchLast;    //!< last touch position in pixels
            SLVec2i         _touch2;       //!< Last finger touch 2 position in pixels
            SLVec2i         _touchDelta;   //!< Delta between two fingers in x
};
//-----------------------------------------------------------------------------
#endif //QTGLWIDGET_H
