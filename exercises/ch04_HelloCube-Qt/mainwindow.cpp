#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>

#include "SL.h"      // Basic SL type definitions
#include "SLImage.h" // Image class for image loading
#include "SLMat4.h"  // 4x4 matrix class
#include "SLVec3.h"  // 3D vector class
#include "glUtils.h" // Basics for OpenGL shaders, buffers & textures
#include "stdafx.h"

using namespace std;

const int VEC_SIZE = 8;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow),
      m_modelViewMatrix(new SLMat4f()), m_projectionMatrix(new SLMat4f()),
      m_viewportMatrix(new SLMat4f()), m_camZ(-2.0f), m_rotAngle(20.0f), m_rotAngle_y(0)

{
    ui->setupUi(this);

    m_v = new SLVec3f *[VEC_SIZE];
    m_v[0] = new SLVec3f(-0.5f, -0.5f, 0.5f);
    m_v[1] = new SLVec3f(0.5f, -0.5f, 0.5f);
    m_v[2] = new SLVec3f(0.5f, 0.5f, 0.5f);
    m_v[3] = new SLVec3f(-0.5f, 0.5f, 0.5f);
    m_v[4] = new SLVec3f(-0.5f, -0.5f, -0.5f);
    m_v[5] = new SLVec3f(0.5f, -0.5f, -0.5f);
    m_v[6] = new SLVec3f(0.5f, 0.5f, -0.5f);
    m_v[7] = new SLVec3f(-0.5f, 0.5f, -0.5f);

    setMouseTracking(true);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);

    QSize size = this->size();
    float width = size.width();
    float height = size.height();
    float aspect = (float)width / (float)height;

    // #1: Create the projection matrix with FOV, aspect, near, far
    // #2: Create the viewport matrix with x, y, viewport-width, vewport-height

    m_projectionMatrix->perspective(50, aspect, 0.1f, 10.0f);
    m_viewportMatrix->viewport(0, 0, width, height, 0, 1);
}

void MainWindow::paintEvent(QPaintEvent *e) {
    QMainWindow::paintEvent(e);

    // #3: Model matrix creation
    // start with identity every frame
    m_modelViewMatrix->identity();

    // view transform: move the coordinate system away from the camera
    m_modelViewMatrix->translate(0, 0, m_camZ);

    // model transform: rotate the coordinate system increasingly
    m_modelViewMatrix->rotate(m_rotAngle += 0.05f, 0, 1, 0);
    //m_modelViewMatrix->rotate(m_rotAngle_y += 0.05f, 1, 0, 0);

    // build combined matrix out of viewport, projection & modelview matrix
    SLMat4f *m = new SLMat4f();
    m->multiply(*m_viewportMatrix);
    m->multiply(*m_projectionMatrix);
    m->multiply(*m_modelViewMatrix); // transform all vertices into screen space
    // (x & y in pixels and z as the depth)
    SLVec3f *v2 = new SLVec3f[VEC_SIZE];
    for (int i = 0; i < VEC_SIZE; ++i) {
        v2[i] = m->multVec(*m_v[i]);
    }

    QPainter painter(this);
    int lineWidth = 1;

    // draw front square
    painter.setPen(QPen(Qt::red, lineWidth, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(v2[0].x, v2[0].y, v2[1].x, v2[1].y);
    painter.drawLine(v2[1].x, v2[1].y, v2[2].x, v2[2].y);
    painter.drawLine(v2[2].x, v2[2].y, v2[3].x, v2[3].y);
    painter.drawLine(v2[3].x, v2[3].y, v2[0].x, v2[0].y);

    // draw back square
    painter.setPen(QPen(Qt::blue, lineWidth, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(v2[4].x, v2[4].y, v2[5].x, v2[5].y);
    painter.drawLine(v2[5].x, v2[5].y, v2[6].x, v2[6].y);
    painter.drawLine(v2[6].x, v2[6].y, v2[7].x, v2[7].y);
    painter.drawLine(v2[7].x, v2[7].y, v2[4].x, v2[4].y);

    // draw from front corners to the back corners
    painter.setPen(QPen(Qt::green, lineWidth, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(v2[0].x, v2[0].y, v2[4].x, v2[4].y);
    painter.drawLine(v2[1].x, v2[1].y, v2[5].x, v2[5].y);
    painter.drawLine(v2[2].x, v2[2].y, v2[6].x, v2[6].y);
    painter.drawLine(v2[3].x, v2[3].y, v2[7].x, v2[7].y);
}

void MainWindow::mouseMoveEvent(QMouseEvent *event){
    QPoint current_mouse = event->pos();

    if (prev_mouse.x() > current_mouse.x())  m_rotAngle -= 2;
    else  m_rotAngle += 2;

    if (prev_mouse.y() > current_mouse.y())  m_rotAngle_y -= 2;
    else  m_rotAngle_y += 2;


    prev_mouse = current_mouse;
    repaint();
}
