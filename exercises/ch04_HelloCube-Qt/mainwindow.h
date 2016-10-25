#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QObject>
#include <QEvent>

#include "SL.h"      // Basic SL type definitions
#include "SLImage.h" // Image class for image loading
#include "SLMat4.h"  // 4x4 matrix class
#include "SLVec3.h"  // 3D vector class
#include "glUtils.h" // Basics for OpenGL shaders, buffers & textures
#include "stdafx.h"

using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = 0);
  void resizeEvent(QResizeEvent *event);
  void paintEvent(QPaintEvent *e);
  void mouseMoveEvent(QMouseEvent *event);
  void update();
  ~MainWindow();

private:
  Ui::MainWindow *ui;
  SLMat4f *m_modelViewMatrix;
  SLMat4f *m_projectionMatrix;
  SLMat4f *m_viewportMatrix;
  SLVec3f **m_v;
  float m_camZ;
  float m_rotAngle;
  float m_rotAngle_y;
  QPoint prev_mouse;
};

#endif // MAINWINDOW_H
