//#############################################################################
//  File:      HelloCube_Qt.h
//  Purpose:   Simple cube rotation demo app without OpenGL using Qt interface
//  Author:    Marcus Hudritsch, Timo Tschanz, Pascal Zingg
//  Date:      Spring 2017
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifndef HELLOCUBE_QT_H
#define HELLOCUBE_QT_H

#include <QMainWindow>
#include <QObject>
#include <QEvent>

#include "SL.h"      // Basic SL type definitions
#include "SLMat4.h"  // 4x4 matrix class
#include "SLVec3.h"  // 3D vector class

using namespace std;

//-----------------------------------------------------------------------------
namespace Ui
{
    class MainWindow;
}
//-----------------------------------------------------------------------------
//! Main window class derived from Qt QMainWindow
class HelloCube_Qt : public QMainWindow
{
    Q_OBJECT

    public:
                HelloCube_Qt    (QWidget *parent = 0);
               ~HelloCube_Qt    ();
        void    resizeEvent     (QResizeEvent *event);
        void    paintEvent      (QPaintEvent *e);
        void    mouseMoveEvent  (QMouseEvent *event);

    private:
        Ui::MainWindow* ui;                 //!< Qt ui window object
        SLMat4f         m_modelViewMatrix;  //!< model & view matrix
        SLMat4f         m_projectionMatrix; //!< projection matrix
        SLMat4f         m_viewportMatrix;   //!< viewport matrix
        SLVec3f         m_v[8];             //!< Array of vertices
        float           m_camZ;             //!< camera distance in z direction
        float           m_rotAngle;         //!< rotation angle in degrees
        float           m_rotAngle_y;
        QPoint          prev_mouse;         //!< previous mouse position
};
//-----------------------------------------------------------------------------

#endif // HelloCube_Qt
