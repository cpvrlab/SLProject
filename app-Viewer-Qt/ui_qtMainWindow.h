/********************************************************************************
** Form generated from reading UI file 'qtMainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.4.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTMAINWINDOW_H
#define UI_QTMAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "qtAnimationSlider.h"
#include "qtPropertyTreeWidget.h"

QT_BEGIN_NAMESPACE

class Ui_qtMainWindow
{
public:
    QAction *action_Quit;
    QAction *actionSmall_Test_Scene;
    QAction *actionLarge_Model;
    QAction *actionFigure;
    QAction *actionMesh_Loader;
    QAction *actionTexture_Blending;
    QAction *actionFrustum_Culling_1;
    QAction *actionFrustum_Culling_2;
    QAction *actionTexture_Filtering;
    QAction *actionPer_Vertex_Lighting;
    QAction *actionPer_Pixel_Lighting;
    QAction *actionPer_Vertex_Wave;
    QAction *actionWater;
    QAction *actionBump_Mapping;
    QAction *actionParallax_Mapping;
    QAction *actionGlass_Shader;
    QAction *actionEarth_Shader;
    QAction *actionRT_Spheres;
    QAction *actionRT_Muttenzer_Box;
    QAction *actionRT_Depth_of_Field;
    QAction *actionRT_Soft_Shadows;
    QAction *actionReset;
    QAction *actionPerspective;
    QAction *actionOrthographic;
    QAction *actionSide_by_side;
    QAction *actionSide_by_side_proportional;
    QAction *actionSide_by_side_distorted;
    QAction *actionLine_by_line;
    QAction *actionColumn_by_column;
    QAction *actionPixel_by_pixel;
    QAction *actionColor_Red_Cyan;
    QAction *actionColor_Red_Green;
    QAction *actionColor_Red_Blue;
    QAction *actionColor_Cyan_Yellow;
    QAction *action_eyeSepInc10;
    QAction *action_eyeSepDec10;
    QAction *action_focalDistInc;
    QAction *action_focalDistDec;
    QAction *action_fovInc10;
    QAction *action_fovDec10;
    QAction *actionTurntable_Y_up;
    QAction *actionTurntable_Z_up;
    QAction *actionWalking_Y_up;
    QAction *actionWalking_Z_up;
    QAction *action_speedInc;
    QAction *action_speedDec;
    QAction *actionAntialiasing;
    QAction *actionView_Frustum_Culling;
    QAction *actionSlowdown_on_Idle;
    QAction *actionShow_Statistics;
    QAction *actionShow_Scene_Info;
    QAction *actionShow_Menu;
    QAction *actionAbout_SLProject;
    QAction *actionCredits;
    QAction *actionAbout_Qt;
    QAction *actionRay_Tracer;
    QAction *actionPath_Tracer;
    QAction *actionOpenGL;
    QAction *actionRender_to_depth_1;
    QAction *actionRender_to_depth_2;
    QAction *actionRender_to_depth_5;
    QAction *actionRender_to_max_depth;
    QAction *actionConstant_Redering;
    QAction *actionRender_Distributed_RT_features;
    QAction *action1_Sample;
    QAction *action10_Samples;
    QAction *action100_Sample;
    QAction *action1000_Samples;
    QAction *action10000_Samples;
    QAction *actionShow_DockScenegraph;
    QAction *actionShow_Statusbar;
    QAction *actionDepthTest;
    QAction *actionShow_Normals;
    QAction *actionShow_Wired_Mesh;
    QAction *actionShow_Bounding_Boxes;
    QAction *actionShow_Axis;
    QAction *actionShow_Backfaces;
    QAction *actionShow_Voxels;
    QAction *actionTextures_off;
    QAction *actionAnimation_off;
    QAction *actionFullscreen;
    QAction *actionShow_DockProperties;
    QAction *actionShow_Toolbar;
    QAction *actionSplit_active_view_horizontally;
    QAction *actionSplit_active_view_vertically;
    QAction *actionDelete_active_view;
    QAction *actionSplit_into_4_views;
    QAction *actionSingle_view;
    QAction *actionMass_Animation;
    QAction *actionUse_SceneView_Camera;
    QAction *actionAnimPlayForward;
    QAction *actionAnimPause;
    QAction *actionAnimPlayBackward;
    QAction *actionAnimNextKeyframe;
    QAction *actionAnimPrevKeyframe;
    QAction *actionAnimSkipToStart;
    QAction *actionAnimSkipToEnd;
    QAction *actionAnimStop;
    QAction *actionShow_Animation_Controler;
    QAction *actionSkeletal_Animation;
    QAction *actionAstroboy_Army_GPU;
    QAction *actionRT_Lens;
    QAction *actionAstroboy_Army_CPU;
    QAction *actionSuperAnimation;
    QWidget *centralWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuLoad_Scene;
    QMenu *menuGeneral_Scenes;
    QMenu *menuShader_Scenes;
    QMenu *menuRay_Tracing_Scenes;
    QMenu *menuAnimation_Scenes;
    QMenu *menuCamera;
    QMenu *menuProjection;
    QMenu *menuStereo;
    QMenu *menuEye_Separation;
    QMenu *menuFocal_Distance;
    QMenu *menuField_of_View;
    QMenu *menuCamAnimation;
    QMenu *menuForward_Speed;
    QMenu *menuRender_Flags;
    QMenu *menuInfos;
    QMenu *menuRenderer;
    QMenu *menuRay_Tracing;
    QMenu *menuPath_tracing;
    QMenu *menuWindow;
    QMenu *menuHelp;
    QMenu *menuAnimation;
    QStatusBar *statusBar;
    QDockWidget *dockScenegraph;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_2;
    QTreeWidget *nodeTree;
    QToolBar *toolBar;
    QDockWidget *dockProperties;
    QWidget *dockWidgetContents_2;
    QVBoxLayout *verticalLayout;
    qtPropertyTreeWidget *propertyTree;
    QDockWidget *dockAnimation;
    QWidget *dockWidgetContents_3;
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout;
    QLabel *animCurrentTimeLabel;
    QAnimationSlider *animTimelineSlider;
    QLabel *animDurationLabel;
    QHBoxLayout *horizontalLayout_4;
    QSpacerItem *horizontalSpacer_2;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_5;
    QComboBox *animAnimatedObjectSelect;
    QComboBox *animAnimationSelect;
    QHBoxLayout *horizontalLayout_6;
    QToolButton *animSkipStartButton;
    QToolButton *animPrevKeyframeButton;
    QToolButton *animPlayBackwardButton;
    QToolButton *animStopButton;
    QToolButton *animPauseButton;
    QToolButton *animPlayForwardButton;
    QToolButton *animNextKeyframeButton;
    QToolButton *animSkipEndButton;
    QDoubleSpinBox *animSpeedInput;
    QDoubleSpinBox *animWeightInput;
    QComboBox *animEasingSelect;
    QComboBox *animLoopingSelect;
    QSpacerItem *horizontalSpacer;

    void setupUi(QMainWindow *qtMainWindow)
    {
        if (qtMainWindow->objectName().isEmpty())
            qtMainWindow->setObjectName(QStringLiteral("qtMainWindow"));
        qtMainWindow->resize(906, 679);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(qtMainWindow->sizePolicy().hasHeightForWidth());
        qtMainWindow->setSizePolicy(sizePolicy);
        qtMainWindow->setMinimumSize(QSize(0, 0));
        qtMainWindow->setFocusPolicy(Qt::NoFocus);
        qtMainWindow->setDockOptions(QMainWindow::AllowNestedDocks|QMainWindow::AllowTabbedDocks|QMainWindow::AnimatedDocks);
        action_Quit = new QAction(qtMainWindow);
        action_Quit->setObjectName(QStringLiteral("action_Quit"));
        actionSmall_Test_Scene = new QAction(qtMainWindow);
        actionSmall_Test_Scene->setObjectName(QStringLiteral("actionSmall_Test_Scene"));
        actionSmall_Test_Scene->setCheckable(true);
        actionLarge_Model = new QAction(qtMainWindow);
        actionLarge_Model->setObjectName(QStringLiteral("actionLarge_Model"));
        actionLarge_Model->setCheckable(true);
        actionFigure = new QAction(qtMainWindow);
        actionFigure->setObjectName(QStringLiteral("actionFigure"));
        actionFigure->setCheckable(true);
        actionMesh_Loader = new QAction(qtMainWindow);
        actionMesh_Loader->setObjectName(QStringLiteral("actionMesh_Loader"));
        actionMesh_Loader->setCheckable(true);
        actionTexture_Blending = new QAction(qtMainWindow);
        actionTexture_Blending->setObjectName(QStringLiteral("actionTexture_Blending"));
        actionTexture_Blending->setCheckable(true);
        actionFrustum_Culling_1 = new QAction(qtMainWindow);
        actionFrustum_Culling_1->setObjectName(QStringLiteral("actionFrustum_Culling_1"));
        actionFrustum_Culling_1->setCheckable(true);
        actionFrustum_Culling_2 = new QAction(qtMainWindow);
        actionFrustum_Culling_2->setObjectName(QStringLiteral("actionFrustum_Culling_2"));
        actionFrustum_Culling_2->setCheckable(true);
        actionTexture_Filtering = new QAction(qtMainWindow);
        actionTexture_Filtering->setObjectName(QStringLiteral("actionTexture_Filtering"));
        actionTexture_Filtering->setCheckable(true);
        actionPer_Vertex_Lighting = new QAction(qtMainWindow);
        actionPer_Vertex_Lighting->setObjectName(QStringLiteral("actionPer_Vertex_Lighting"));
        actionPer_Vertex_Lighting->setCheckable(true);
        actionPer_Pixel_Lighting = new QAction(qtMainWindow);
        actionPer_Pixel_Lighting->setObjectName(QStringLiteral("actionPer_Pixel_Lighting"));
        actionPer_Pixel_Lighting->setCheckable(true);
        actionPer_Vertex_Wave = new QAction(qtMainWindow);
        actionPer_Vertex_Wave->setObjectName(QStringLiteral("actionPer_Vertex_Wave"));
        actionPer_Vertex_Wave->setCheckable(true);
        actionWater = new QAction(qtMainWindow);
        actionWater->setObjectName(QStringLiteral("actionWater"));
        actionWater->setCheckable(true);
        actionBump_Mapping = new QAction(qtMainWindow);
        actionBump_Mapping->setObjectName(QStringLiteral("actionBump_Mapping"));
        actionBump_Mapping->setCheckable(true);
        actionParallax_Mapping = new QAction(qtMainWindow);
        actionParallax_Mapping->setObjectName(QStringLiteral("actionParallax_Mapping"));
        actionParallax_Mapping->setCheckable(true);
        actionGlass_Shader = new QAction(qtMainWindow);
        actionGlass_Shader->setObjectName(QStringLiteral("actionGlass_Shader"));
        actionGlass_Shader->setCheckable(true);
        actionEarth_Shader = new QAction(qtMainWindow);
        actionEarth_Shader->setObjectName(QStringLiteral("actionEarth_Shader"));
        actionEarth_Shader->setCheckable(true);
        actionRT_Spheres = new QAction(qtMainWindow);
        actionRT_Spheres->setObjectName(QStringLiteral("actionRT_Spheres"));
        actionRT_Spheres->setCheckable(true);
        actionRT_Muttenzer_Box = new QAction(qtMainWindow);
        actionRT_Muttenzer_Box->setObjectName(QStringLiteral("actionRT_Muttenzer_Box"));
        actionRT_Muttenzer_Box->setCheckable(true);
        actionRT_Depth_of_Field = new QAction(qtMainWindow);
        actionRT_Depth_of_Field->setObjectName(QStringLiteral("actionRT_Depth_of_Field"));
        actionRT_Depth_of_Field->setCheckable(true);
        actionRT_Soft_Shadows = new QAction(qtMainWindow);
        actionRT_Soft_Shadows->setObjectName(QStringLiteral("actionRT_Soft_Shadows"));
        actionRT_Soft_Shadows->setCheckable(true);
        actionReset = new QAction(qtMainWindow);
        actionReset->setObjectName(QStringLiteral("actionReset"));
        actionPerspective = new QAction(qtMainWindow);
        actionPerspective->setObjectName(QStringLiteral("actionPerspective"));
        actionPerspective->setCheckable(true);
        QIcon icon;
        icon.addFile(QStringLiteral(":/images/ProjectionPersp.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPerspective->setIcon(icon);
        actionPerspective->setShortcutContext(Qt::ApplicationShortcut);
        actionOrthographic = new QAction(qtMainWindow);
        actionOrthographic->setObjectName(QStringLiteral("actionOrthographic"));
        actionOrthographic->setCheckable(true);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/images/ProjectionOrtho.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionOrthographic->setIcon(icon1);
        actionOrthographic->setShortcutContext(Qt::ApplicationShortcut);
        actionSide_by_side = new QAction(qtMainWindow);
        actionSide_by_side->setObjectName(QStringLiteral("actionSide_by_side"));
        actionSide_by_side->setCheckable(true);
        actionSide_by_side_proportional = new QAction(qtMainWindow);
        actionSide_by_side_proportional->setObjectName(QStringLiteral("actionSide_by_side_proportional"));
        actionSide_by_side_proportional->setCheckable(true);
        actionSide_by_side_distorted = new QAction(qtMainWindow);
        actionSide_by_side_distorted->setObjectName(QStringLiteral("actionSide_by_side_distorted"));
        actionSide_by_side_distorted->setCheckable(true);
        actionLine_by_line = new QAction(qtMainWindow);
        actionLine_by_line->setObjectName(QStringLiteral("actionLine_by_line"));
        actionLine_by_line->setCheckable(true);
        actionColumn_by_column = new QAction(qtMainWindow);
        actionColumn_by_column->setObjectName(QStringLiteral("actionColumn_by_column"));
        actionColumn_by_column->setCheckable(true);
        actionPixel_by_pixel = new QAction(qtMainWindow);
        actionPixel_by_pixel->setObjectName(QStringLiteral("actionPixel_by_pixel"));
        actionPixel_by_pixel->setCheckable(true);
        actionColor_Red_Cyan = new QAction(qtMainWindow);
        actionColor_Red_Cyan->setObjectName(QStringLiteral("actionColor_Red_Cyan"));
        actionColor_Red_Cyan->setCheckable(true);
        actionColor_Red_Green = new QAction(qtMainWindow);
        actionColor_Red_Green->setObjectName(QStringLiteral("actionColor_Red_Green"));
        actionColor_Red_Green->setCheckable(true);
        actionColor_Red_Blue = new QAction(qtMainWindow);
        actionColor_Red_Blue->setObjectName(QStringLiteral("actionColor_Red_Blue"));
        actionColor_Red_Blue->setCheckable(true);
        actionColor_Cyan_Yellow = new QAction(qtMainWindow);
        actionColor_Cyan_Yellow->setObjectName(QStringLiteral("actionColor_Cyan_Yellow"));
        actionColor_Cyan_Yellow->setCheckable(true);
        action_eyeSepInc10 = new QAction(qtMainWindow);
        action_eyeSepInc10->setObjectName(QStringLiteral("action_eyeSepInc10"));
        action_eyeSepDec10 = new QAction(qtMainWindow);
        action_eyeSepDec10->setObjectName(QStringLiteral("action_eyeSepDec10"));
        action_focalDistInc = new QAction(qtMainWindow);
        action_focalDistInc->setObjectName(QStringLiteral("action_focalDistInc"));
        action_focalDistDec = new QAction(qtMainWindow);
        action_focalDistDec->setObjectName(QStringLiteral("action_focalDistDec"));
        action_fovInc10 = new QAction(qtMainWindow);
        action_fovInc10->setObjectName(QStringLiteral("action_fovInc10"));
        action_fovInc10->setShortcutContext(Qt::ApplicationShortcut);
        action_fovDec10 = new QAction(qtMainWindow);
        action_fovDec10->setObjectName(QStringLiteral("action_fovDec10"));
        action_fovDec10->setShortcutContext(Qt::ApplicationShortcut);
        actionTurntable_Y_up = new QAction(qtMainWindow);
        actionTurntable_Y_up->setObjectName(QStringLiteral("actionTurntable_Y_up"));
        actionTurntable_Y_up->setCheckable(true);
        actionTurntable_Y_up->setChecked(true);
        actionTurntable_Z_up = new QAction(qtMainWindow);
        actionTurntable_Z_up->setObjectName(QStringLiteral("actionTurntable_Z_up"));
        actionTurntable_Z_up->setCheckable(true);
        actionWalking_Y_up = new QAction(qtMainWindow);
        actionWalking_Y_up->setObjectName(QStringLiteral("actionWalking_Y_up"));
        actionWalking_Y_up->setCheckable(true);
        actionWalking_Z_up = new QAction(qtMainWindow);
        actionWalking_Z_up->setObjectName(QStringLiteral("actionWalking_Z_up"));
        actionWalking_Z_up->setCheckable(true);
        action_speedInc = new QAction(qtMainWindow);
        action_speedInc->setObjectName(QStringLiteral("action_speedInc"));
        action_speedDec = new QAction(qtMainWindow);
        action_speedDec->setObjectName(QStringLiteral("action_speedDec"));
        actionAntialiasing = new QAction(qtMainWindow);
        actionAntialiasing->setObjectName(QStringLiteral("actionAntialiasing"));
        actionAntialiasing->setCheckable(true);
        actionAntialiasing->setChecked(true);
        actionView_Frustum_Culling = new QAction(qtMainWindow);
        actionView_Frustum_Culling->setObjectName(QStringLiteral("actionView_Frustum_Culling"));
        actionView_Frustum_Culling->setCheckable(true);
        actionView_Frustum_Culling->setChecked(true);
        actionSlowdown_on_Idle = new QAction(qtMainWindow);
        actionSlowdown_on_Idle->setObjectName(QStringLiteral("actionSlowdown_on_Idle"));
        actionSlowdown_on_Idle->setCheckable(true);
        actionSlowdown_on_Idle->setChecked(true);
        actionShow_Statistics = new QAction(qtMainWindow);
        actionShow_Statistics->setObjectName(QStringLiteral("actionShow_Statistics"));
        actionShow_Statistics->setCheckable(true);
        actionShow_Scene_Info = new QAction(qtMainWindow);
        actionShow_Scene_Info->setObjectName(QStringLiteral("actionShow_Scene_Info"));
        actionShow_Scene_Info->setCheckable(true);
        actionShow_Scene_Info->setChecked(false);
        actionShow_Menu = new QAction(qtMainWindow);
        actionShow_Menu->setObjectName(QStringLiteral("actionShow_Menu"));
        actionShow_Menu->setCheckable(true);
        actionAbout_SLProject = new QAction(qtMainWindow);
        actionAbout_SLProject->setObjectName(QStringLiteral("actionAbout_SLProject"));
        actionCredits = new QAction(qtMainWindow);
        actionCredits->setObjectName(QStringLiteral("actionCredits"));
        actionAbout_Qt = new QAction(qtMainWindow);
        actionAbout_Qt->setObjectName(QStringLiteral("actionAbout_Qt"));
        actionRay_Tracer = new QAction(qtMainWindow);
        actionRay_Tracer->setObjectName(QStringLiteral("actionRay_Tracer"));
        actionRay_Tracer->setCheckable(true);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/images/renderRT.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionRay_Tracer->setIcon(icon2);
        actionPath_Tracer = new QAction(qtMainWindow);
        actionPath_Tracer->setObjectName(QStringLiteral("actionPath_Tracer"));
        actionPath_Tracer->setCheckable(true);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/images/renderPT.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPath_Tracer->setIcon(icon3);
        actionOpenGL = new QAction(qtMainWindow);
        actionOpenGL->setObjectName(QStringLiteral("actionOpenGL"));
        actionOpenGL->setCheckable(true);
        actionOpenGL->setChecked(true);
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/images/renderGL.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenGL->setIcon(icon4);
        actionRender_to_depth_1 = new QAction(qtMainWindow);
        actionRender_to_depth_1->setObjectName(QStringLiteral("actionRender_to_depth_1"));
        actionRender_to_depth_1->setEnabled(true);
        actionRender_to_depth_2 = new QAction(qtMainWindow);
        actionRender_to_depth_2->setObjectName(QStringLiteral("actionRender_to_depth_2"));
        actionRender_to_depth_5 = new QAction(qtMainWindow);
        actionRender_to_depth_5->setObjectName(QStringLiteral("actionRender_to_depth_5"));
        actionRender_to_max_depth = new QAction(qtMainWindow);
        actionRender_to_max_depth->setObjectName(QStringLiteral("actionRender_to_max_depth"));
        actionConstant_Redering = new QAction(qtMainWindow);
        actionConstant_Redering->setObjectName(QStringLiteral("actionConstant_Redering"));
        actionConstant_Redering->setCheckable(true);
        actionRender_Distributed_RT_features = new QAction(qtMainWindow);
        actionRender_Distributed_RT_features->setObjectName(QStringLiteral("actionRender_Distributed_RT_features"));
        actionRender_Distributed_RT_features->setCheckable(true);
        action1_Sample = new QAction(qtMainWindow);
        action1_Sample->setObjectName(QStringLiteral("action1_Sample"));
        action10_Samples = new QAction(qtMainWindow);
        action10_Samples->setObjectName(QStringLiteral("action10_Samples"));
        action100_Sample = new QAction(qtMainWindow);
        action100_Sample->setObjectName(QStringLiteral("action100_Sample"));
        action1000_Samples = new QAction(qtMainWindow);
        action1000_Samples->setObjectName(QStringLiteral("action1000_Samples"));
        action10000_Samples = new QAction(qtMainWindow);
        action10000_Samples->setObjectName(QStringLiteral("action10000_Samples"));
        actionShow_DockScenegraph = new QAction(qtMainWindow);
        actionShow_DockScenegraph->setObjectName(QStringLiteral("actionShow_DockScenegraph"));
        actionShow_DockScenegraph->setCheckable(true);
        actionShow_DockScenegraph->setChecked(true);
        actionShow_Statusbar = new QAction(qtMainWindow);
        actionShow_Statusbar->setObjectName(QStringLiteral("actionShow_Statusbar"));
        actionShow_Statusbar->setCheckable(true);
        actionShow_Statusbar->setChecked(true);
        actionDepthTest = new QAction(qtMainWindow);
        actionDepthTest->setObjectName(QStringLiteral("actionDepthTest"));
        actionDepthTest->setCheckable(true);
        actionDepthTest->setChecked(true);
        actionShow_Normals = new QAction(qtMainWindow);
        actionShow_Normals->setObjectName(QStringLiteral("actionShow_Normals"));
        actionShow_Normals->setCheckable(true);
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/images/showNormals.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Normals->setIcon(icon5);
        actionShow_Wired_Mesh = new QAction(qtMainWindow);
        actionShow_Wired_Mesh->setObjectName(QStringLiteral("actionShow_Wired_Mesh"));
        actionShow_Wired_Mesh->setCheckable(true);
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/images/showWiremesh.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Wired_Mesh->setIcon(icon6);
        actionShow_Bounding_Boxes = new QAction(qtMainWindow);
        actionShow_Bounding_Boxes->setObjectName(QStringLiteral("actionShow_Bounding_Boxes"));
        actionShow_Bounding_Boxes->setCheckable(true);
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/images/showAABB.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Bounding_Boxes->setIcon(icon7);
        actionShow_Axis = new QAction(qtMainWindow);
        actionShow_Axis->setObjectName(QStringLiteral("actionShow_Axis"));
        actionShow_Axis->setCheckable(true);
        QIcon icon8;
        icon8.addFile(QStringLiteral(":/images/showAxis.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Axis->setIcon(icon8);
        actionShow_Backfaces = new QAction(qtMainWindow);
        actionShow_Backfaces->setObjectName(QStringLiteral("actionShow_Backfaces"));
        actionShow_Backfaces->setCheckable(true);
        QIcon icon9;
        icon9.addFile(QStringLiteral(":/images/showBackfaces.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Backfaces->setIcon(icon9);
        actionShow_Voxels = new QAction(qtMainWindow);
        actionShow_Voxels->setObjectName(QStringLiteral("actionShow_Voxels"));
        actionShow_Voxels->setCheckable(true);
        QIcon icon10;
        icon10.addFile(QStringLiteral(":/images/showVoxels.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Voxels->setIcon(icon10);
        actionTextures_off = new QAction(qtMainWindow);
        actionTextures_off->setObjectName(QStringLiteral("actionTextures_off"));
        actionTextures_off->setCheckable(true);
        QIcon icon11;
        icon11.addFile(QStringLiteral(":/images/noTextures.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTextures_off->setIcon(icon11);
        actionAnimation_off = new QAction(qtMainWindow);
        actionAnimation_off->setObjectName(QStringLiteral("actionAnimation_off"));
        actionAnimation_off->setCheckable(true);
        QIcon icon12;
        icon12.addFile(QStringLiteral(":/images/stopAnimation.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimation_off->setIcon(icon12);
        actionFullscreen = new QAction(qtMainWindow);
        actionFullscreen->setObjectName(QStringLiteral("actionFullscreen"));
        actionFullscreen->setCheckable(true);
        actionShow_DockProperties = new QAction(qtMainWindow);
        actionShow_DockProperties->setObjectName(QStringLiteral("actionShow_DockProperties"));
        actionShow_DockProperties->setCheckable(true);
        actionShow_DockProperties->setChecked(true);
        actionShow_Toolbar = new QAction(qtMainWindow);
        actionShow_Toolbar->setObjectName(QStringLiteral("actionShow_Toolbar"));
        actionShow_Toolbar->setCheckable(true);
        actionShow_Toolbar->setChecked(true);
        actionSplit_active_view_horizontally = new QAction(qtMainWindow);
        actionSplit_active_view_horizontally->setObjectName(QStringLiteral("actionSplit_active_view_horizontally"));
        QIcon icon13;
        icon13.addFile(QStringLiteral(":/images/splitHorizontal.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSplit_active_view_horizontally->setIcon(icon13);
        actionSplit_active_view_vertically = new QAction(qtMainWindow);
        actionSplit_active_view_vertically->setObjectName(QStringLiteral("actionSplit_active_view_vertically"));
        QIcon icon14;
        icon14.addFile(QStringLiteral(":/images/splitVertical.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSplit_active_view_vertically->setIcon(icon14);
        actionDelete_active_view = new QAction(qtMainWindow);
        actionDelete_active_view->setObjectName(QStringLiteral("actionDelete_active_view"));
        actionDelete_active_view->setEnabled(true);
        QIcon icon15;
        icon15.addFile(QStringLiteral(":/images/deleteView.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionDelete_active_view->setIcon(icon15);
        actionSplit_into_4_views = new QAction(qtMainWindow);
        actionSplit_into_4_views->setObjectName(QStringLiteral("actionSplit_into_4_views"));
        QIcon icon16;
        icon16.addFile(QStringLiteral(":/images/splitInto4Views.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSplit_into_4_views->setIcon(icon16);
        actionSingle_view = new QAction(qtMainWindow);
        actionSingle_view->setObjectName(QStringLiteral("actionSingle_view"));
        QIcon icon17;
        icon17.addFile(QStringLiteral(":/images/singleView.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSingle_view->setIcon(icon17);
        actionMass_Animation = new QAction(qtMainWindow);
        actionMass_Animation->setObjectName(QStringLiteral("actionMass_Animation"));
        actionMass_Animation->setCheckable(true);
        actionUse_SceneView_Camera = new QAction(qtMainWindow);
        actionUse_SceneView_Camera->setObjectName(QStringLiteral("actionUse_SceneView_Camera"));
        actionUse_SceneView_Camera->setCheckable(true);
        actionAnimPlayForward = new QAction(qtMainWindow);
        actionAnimPlayForward->setObjectName(QStringLiteral("actionAnimPlayForward"));
        QIcon icon18;
        icon18.addFile(QStringLiteral("images/animPlay.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimPlayForward->setIcon(icon18);
        actionAnimPause = new QAction(qtMainWindow);
        actionAnimPause->setObjectName(QStringLiteral("actionAnimPause"));
        QIcon icon19;
        icon19.addFile(QStringLiteral("images/animPause.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimPause->setIcon(icon19);
        actionAnimPlayBackward = new QAction(qtMainWindow);
        actionAnimPlayBackward->setObjectName(QStringLiteral("actionAnimPlayBackward"));
        QIcon icon20;
        icon20.addFile(QStringLiteral("images/animPlayBackwards.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimPlayBackward->setIcon(icon20);
        actionAnimNextKeyframe = new QAction(qtMainWindow);
        actionAnimNextKeyframe->setObjectName(QStringLiteral("actionAnimNextKeyframe"));
        QIcon icon21;
        icon21.addFile(QStringLiteral("images/animNextKF.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimNextKeyframe->setIcon(icon21);
        actionAnimPrevKeyframe = new QAction(qtMainWindow);
        actionAnimPrevKeyframe->setObjectName(QStringLiteral("actionAnimPrevKeyframe"));
        QIcon icon22;
        icon22.addFile(QStringLiteral("images/animPrevKF.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimPrevKeyframe->setIcon(icon22);
        actionAnimSkipToStart = new QAction(qtMainWindow);
        actionAnimSkipToStart->setObjectName(QStringLiteral("actionAnimSkipToStart"));
        QIcon icon23;
        icon23.addFile(QStringLiteral("images/animGoToStart.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimSkipToStart->setIcon(icon23);
        actionAnimSkipToEnd = new QAction(qtMainWindow);
        actionAnimSkipToEnd->setObjectName(QStringLiteral("actionAnimSkipToEnd"));
        QIcon icon24;
        icon24.addFile(QStringLiteral("images/animGoToEnd.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimSkipToEnd->setIcon(icon24);
        actionAnimStop = new QAction(qtMainWindow);
        actionAnimStop->setObjectName(QStringLiteral("actionAnimStop"));
        QIcon icon25;
        icon25.addFile(QStringLiteral("images/animStop.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAnimStop->setIcon(icon25);
        actionShow_Animation_Controler = new QAction(qtMainWindow);
        actionShow_Animation_Controler->setObjectName(QStringLiteral("actionShow_Animation_Controler"));
        actionShow_Animation_Controler->setCheckable(true);
        actionShow_Animation_Controler->setChecked(true);
        actionSkeletal_Animation = new QAction(qtMainWindow);
        actionSkeletal_Animation->setObjectName(QStringLiteral("actionSkeletal_Animation"));
        actionSkeletal_Animation->setCheckable(true);
        actionAstroboy_Army_GPU = new QAction(qtMainWindow);
        actionAstroboy_Army_GPU->setObjectName(QStringLiteral("actionAstroboy_Army_GPU"));
        actionAstroboy_Army_GPU->setCheckable(true);
        actionRT_Lens = new QAction(qtMainWindow);
        actionRT_Lens->setObjectName(QStringLiteral("actionRT_Lens"));
        actionRT_Lens->setCheckable(true);
        actionAstroboy_Army_CPU = new QAction(qtMainWindow);
        actionAstroboy_Army_CPU->setObjectName(QStringLiteral("actionAstroboy_Army_CPU"));
        actionAstroboy_Army_CPU->setCheckable(true);
        actionSuperAnimation = new QAction(qtMainWindow);
        actionSuperAnimation->setObjectName(QStringLiteral("actionSuperAnimation"));
        centralWidget = new QWidget(qtMainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        qtMainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(qtMainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 906, 31));
        menuBar->setDefaultUp(false);
        menuBar->setNativeMenuBar(false);
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuFile->setToolTipsVisible(true);
        menuLoad_Scene = new QMenu(menuFile);
        menuLoad_Scene->setObjectName(QStringLiteral("menuLoad_Scene"));
        menuGeneral_Scenes = new QMenu(menuLoad_Scene);
        menuGeneral_Scenes->setObjectName(QStringLiteral("menuGeneral_Scenes"));
        menuShader_Scenes = new QMenu(menuLoad_Scene);
        menuShader_Scenes->setObjectName(QStringLiteral("menuShader_Scenes"));
        menuRay_Tracing_Scenes = new QMenu(menuLoad_Scene);
        menuRay_Tracing_Scenes->setObjectName(QStringLiteral("menuRay_Tracing_Scenes"));
        menuAnimation_Scenes = new QMenu(menuLoad_Scene);
        menuAnimation_Scenes->setObjectName(QStringLiteral("menuAnimation_Scenes"));
        menuCamera = new QMenu(menuBar);
        menuCamera->setObjectName(QStringLiteral("menuCamera"));
        menuProjection = new QMenu(menuCamera);
        menuProjection->setObjectName(QStringLiteral("menuProjection"));
        menuStereo = new QMenu(menuProjection);
        menuStereo->setObjectName(QStringLiteral("menuStereo"));
        menuEye_Separation = new QMenu(menuStereo);
        menuEye_Separation->setObjectName(QStringLiteral("menuEye_Separation"));
        menuFocal_Distance = new QMenu(menuStereo);
        menuFocal_Distance->setObjectName(QStringLiteral("menuFocal_Distance"));
        menuField_of_View = new QMenu(menuCamera);
        menuField_of_View->setObjectName(QStringLiteral("menuField_of_View"));
        menuCamAnimation = new QMenu(menuCamera);
        menuCamAnimation->setObjectName(QStringLiteral("menuCamAnimation"));
        menuForward_Speed = new QMenu(menuCamera);
        menuForward_Speed->setObjectName(QStringLiteral("menuForward_Speed"));
        menuRender_Flags = new QMenu(menuBar);
        menuRender_Flags->setObjectName(QStringLiteral("menuRender_Flags"));
        menuInfos = new QMenu(menuBar);
        menuInfos->setObjectName(QStringLiteral("menuInfos"));
        menuRenderer = new QMenu(menuBar);
        menuRenderer->setObjectName(QStringLiteral("menuRenderer"));
        menuRay_Tracing = new QMenu(menuBar);
        menuRay_Tracing->setObjectName(QStringLiteral("menuRay_Tracing"));
        menuRay_Tracing->setEnabled(true);
        menuPath_tracing = new QMenu(menuBar);
        menuPath_tracing->setObjectName(QStringLiteral("menuPath_tracing"));
        menuWindow = new QMenu(menuBar);
        menuWindow->setObjectName(QStringLiteral("menuWindow"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        menuAnimation = new QMenu(menuBar);
        menuAnimation->setObjectName(QStringLiteral("menuAnimation"));
        qtMainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(qtMainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        qtMainWindow->setStatusBar(statusBar);
        dockScenegraph = new QDockWidget(qtMainWindow);
        dockScenegraph->setObjectName(QStringLiteral("dockScenegraph"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(dockScenegraph->sizePolicy().hasHeightForWidth());
        dockScenegraph->setSizePolicy(sizePolicy1);
        dockScenegraph->setMinimumSize(QSize(160, 138));
        dockScenegraph->setFloating(false);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        dockWidgetContents->setLayoutDirection(Qt::LeftToRight);
        dockWidgetContents->setAutoFillBackground(false);
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_2->setSpacing(0);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        nodeTree = new QTreeWidget(dockWidgetContents);
        QFont font;
        font.setPointSize(10);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QStringLiteral("1"));
        __qtreewidgetitem->setFont(0, font);
        nodeTree->setHeaderItem(__qtreewidgetitem);
        nodeTree->setObjectName(QStringLiteral("nodeTree"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(nodeTree->sizePolicy().hasHeightForWidth());
        nodeTree->setSizePolicy(sizePolicy2);
        nodeTree->setMaximumSize(QSize(16777215, 16777215));
        nodeTree->setFont(font);
        nodeTree->setAutoScrollMargin(10);
        nodeTree->setAlternatingRowColors(true);
        nodeTree->setSelectionMode(QAbstractItemView::MultiSelection);
        nodeTree->setIndentation(16);
        nodeTree->setAnimated(false);
        nodeTree->setColumnCount(1);
        nodeTree->header()->setVisible(false);

        verticalLayout_2->addWidget(nodeTree);

        dockScenegraph->setWidget(dockWidgetContents);
        qtMainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockScenegraph);
        toolBar = new QToolBar(qtMainWindow);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        toolBar->setIconSize(QSize(24, 24));
        qtMainWindow->addToolBar(Qt::TopToolBarArea, toolBar);
        dockProperties = new QDockWidget(qtMainWindow);
        dockProperties->setObjectName(QStringLiteral("dockProperties"));
        dockProperties->setMinimumSize(QSize(160, 138));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QStringLiteral("dockWidgetContents_2"));
        verticalLayout = new QVBoxLayout(dockWidgetContents_2);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        propertyTree = new qtPropertyTreeWidget(dockWidgetContents_2);
        propertyTree->setObjectName(QStringLiteral("propertyTree"));
        propertyTree->setFont(font);
        propertyTree->setAlternatingRowColors(true);
        propertyTree->setIndentation(16);
        propertyTree->setRootIsDecorated(true);
        propertyTree->setUniformRowHeights(false);
        propertyTree->setAnimated(true);
        propertyTree->setHeaderHidden(true);
        propertyTree->setColumnCount(2);
        propertyTree->header()->setVisible(false);
        propertyTree->header()->setCascadingSectionResizes(false);
        propertyTree->header()->setDefaultSectionSize(150);
        propertyTree->header()->setMinimumSectionSize(50);
        propertyTree->header()->setStretchLastSection(true);

        verticalLayout->addWidget(propertyTree);

        dockProperties->setWidget(dockWidgetContents_2);
        qtMainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockProperties);
        dockAnimation = new QDockWidget(qtMainWindow);
        dockAnimation->setObjectName(QStringLiteral("dockAnimation"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(dockAnimation->sizePolicy().hasHeightForWidth());
        dockAnimation->setSizePolicy(sizePolicy3);
        dockAnimation->setMinimumSize(QSize(643, 126));
        dockAnimation->setFloating(false);
        dockAnimation->setFeatures(QDockWidget::AllDockWidgetFeatures);
        dockWidgetContents_3 = new QWidget();
        dockWidgetContents_3->setObjectName(QStringLiteral("dockWidgetContents_3"));
        sizePolicy.setHeightForWidth(dockWidgetContents_3->sizePolicy().hasHeightForWidth());
        dockWidgetContents_3->setSizePolicy(sizePolicy);
        verticalLayout_3 = new QVBoxLayout(dockWidgetContents_3);
        verticalLayout_3->setSpacing(2);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(4, 2, 4, 2);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(10);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(3, 0, 3, -1);
        animCurrentTimeLabel = new QLabel(dockWidgetContents_3);
        animCurrentTimeLabel->setObjectName(QStringLiteral("animCurrentTimeLabel"));
        QSizePolicy sizePolicy4(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(animCurrentTimeLabel->sizePolicy().hasHeightForWidth());
        animCurrentTimeLabel->setSizePolicy(sizePolicy4);
        animCurrentTimeLabel->setMinimumSize(QSize(20, 0));
        animCurrentTimeLabel->setFont(font);
        animCurrentTimeLabel->setLayoutDirection(Qt::LeftToRight);
        animCurrentTimeLabel->setIndent(0);

        horizontalLayout->addWidget(animCurrentTimeLabel);

        animTimelineSlider = new QAnimationSlider(dockWidgetContents_3);
        animTimelineSlider->setObjectName(QStringLiteral("animTimelineSlider"));
        QSizePolicy sizePolicy5(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(animTimelineSlider->sizePolicy().hasHeightForWidth());
        animTimelineSlider->setSizePolicy(sizePolicy5);
        animTimelineSlider->setMouseTracking(true);
        animTimelineSlider->setStyleSheet(QLatin1String("QSlider::groove:horizontal {\n"
"	border: 1px solid #bbb;\n"
"	background: white;\n"
"	height: 10px;\n"
"	border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"    	stop:0 #EDA215, stop:1 #C98300);\n"
"	border: 1px solid #bbb;\n"
"	height: 10px;\n"
"	border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"		stop:0 #cdcdcd, stop:1 #fafafa);\n"
"	border: 1px solid #ddd;\n"
"	height: 10px;\n"
"	border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"	    stop:0 #fff, stop:1 #ccc);\n"
"	border: 1px solid #bbb;\n"
"	width: 15px;\n"
"	margin-top: -3px;\n"
"	margin-bottom: -3px;\n"
"	border-radius:8px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"    	stop:0 #fff, stop:1 #ddd);\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal:disabled {\n"
"	backgr"
                        "ound: #bbb;\n"
"	border-color: #999;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal:disabled {\n"
"	background: #eee;\n"
"	border-color: #999;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:disabled {\n"
"	background: #eee;\n"
"	border: 1px solid #aaa;\n"
"	border-radius: 4px;\n"
"}"));
        animTimelineSlider->setMaximum(999);
        animTimelineSlider->setSingleStep(10);
        animTimelineSlider->setPageStep(100);
        animTimelineSlider->setValue(0);
        animTimelineSlider->setSliderPosition(0);
        animTimelineSlider->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(animTimelineSlider);

        animDurationLabel = new QLabel(dockWidgetContents_3);
        animDurationLabel->setObjectName(QStringLiteral("animDurationLabel"));
        sizePolicy4.setHeightForWidth(animDurationLabel->sizePolicy().hasHeightForWidth());
        animDurationLabel->setSizePolicy(sizePolicy4);
        animDurationLabel->setMinimumSize(QSize(20, 0));
        animDurationLabel->setFont(font);
        animDurationLabel->setMargin(0);
        animDurationLabel->setIndent(0);

        horizontalLayout->addWidget(animDurationLabel);


        verticalLayout_3->addLayout(horizontalLayout);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalSpacer_2 = new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_2);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, -1, 0, 0);
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(1);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, -1, -1, -1);
        animAnimatedObjectSelect = new QComboBox(dockWidgetContents_3);
        animAnimatedObjectSelect->setObjectName(QStringLiteral("animAnimatedObjectSelect"));

        horizontalLayout_5->addWidget(animAnimatedObjectSelect);

        animAnimationSelect = new QComboBox(dockWidgetContents_3);
        animAnimationSelect->setObjectName(QStringLiteral("animAnimationSelect"));

        horizontalLayout_5->addWidget(animAnimationSelect);


        verticalLayout_5->addLayout(horizontalLayout_5);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(1);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        animSkipStartButton = new QToolButton(dockWidgetContents_3);
        animSkipStartButton->setObjectName(QStringLiteral("animSkipStartButton"));
        QIcon icon26;
        icon26.addFile(QStringLiteral(":/images/animGoToStart.png"), QSize(), QIcon::Normal, QIcon::Off);
        animSkipStartButton->setIcon(icon26);

        horizontalLayout_6->addWidget(animSkipStartButton);

        animPrevKeyframeButton = new QToolButton(dockWidgetContents_3);
        animPrevKeyframeButton->setObjectName(QStringLiteral("animPrevKeyframeButton"));
        QIcon icon27;
        icon27.addFile(QStringLiteral(":/images/animPrevKF.png"), QSize(), QIcon::Normal, QIcon::Off);
        animPrevKeyframeButton->setIcon(icon27);

        horizontalLayout_6->addWidget(animPrevKeyframeButton);

        animPlayBackwardButton = new QToolButton(dockWidgetContents_3);
        animPlayBackwardButton->setObjectName(QStringLiteral("animPlayBackwardButton"));
        QIcon icon28;
        icon28.addFile(QStringLiteral(":/images/animPlayBackwards.png"), QSize(), QIcon::Normal, QIcon::Off);
        animPlayBackwardButton->setIcon(icon28);

        horizontalLayout_6->addWidget(animPlayBackwardButton);

        animStopButton = new QToolButton(dockWidgetContents_3);
        animStopButton->setObjectName(QStringLiteral("animStopButton"));
        QIcon icon29;
        icon29.addFile(QStringLiteral(":/images/animStop.png"), QSize(), QIcon::Normal, QIcon::Off);
        animStopButton->setIcon(icon29);

        horizontalLayout_6->addWidget(animStopButton);

        animPauseButton = new QToolButton(dockWidgetContents_3);
        animPauseButton->setObjectName(QStringLiteral("animPauseButton"));
        QIcon icon30;
        icon30.addFile(QStringLiteral(":/images/animPause.png"), QSize(), QIcon::Normal, QIcon::Off);
        animPauseButton->setIcon(icon30);

        horizontalLayout_6->addWidget(animPauseButton);

        animPlayForwardButton = new QToolButton(dockWidgetContents_3);
        animPlayForwardButton->setObjectName(QStringLiteral("animPlayForwardButton"));
        QIcon icon31;
        icon31.addFile(QStringLiteral(":/images/animPlay.png"), QSize(), QIcon::Normal, QIcon::Off);
        animPlayForwardButton->setIcon(icon31);

        horizontalLayout_6->addWidget(animPlayForwardButton);

        animNextKeyframeButton = new QToolButton(dockWidgetContents_3);
        animNextKeyframeButton->setObjectName(QStringLiteral("animNextKeyframeButton"));
        QIcon icon32;
        icon32.addFile(QStringLiteral(":/images/animNextKF.png"), QSize(), QIcon::Normal, QIcon::Off);
        animNextKeyframeButton->setIcon(icon32);

        horizontalLayout_6->addWidget(animNextKeyframeButton);

        animSkipEndButton = new QToolButton(dockWidgetContents_3);
        animSkipEndButton->setObjectName(QStringLiteral("animSkipEndButton"));
        QIcon icon33;
        icon33.addFile(QStringLiteral(":/images/animGoToEnd.png"), QSize(), QIcon::Normal, QIcon::Off);
        animSkipEndButton->setIcon(icon33);

        horizontalLayout_6->addWidget(animSkipEndButton);

        animSpeedInput = new QDoubleSpinBox(dockWidgetContents_3);
        animSpeedInput->setObjectName(QStringLiteral("animSpeedInput"));
        animSpeedInput->setEnabled(true);
        sizePolicy4.setHeightForWidth(animSpeedInput->sizePolicy().hasHeightForWidth());
        animSpeedInput->setSizePolicy(sizePolicy4);
        animSpeedInput->setMinimumSize(QSize(40, 0));
        animSpeedInput->setMaximumSize(QSize(43, 16777215));
        QFont font1;
        font1.setFamily(QStringLiteral("MS Shell Dlg 2"));
        font1.setPointSize(8);
        animSpeedInput->setFont(font1);
        animSpeedInput->setMaximum(20);
        animSpeedInput->setSingleStep(0.1);
        animSpeedInput->setValue(1);

        horizontalLayout_6->addWidget(animSpeedInput);

        animWeightInput = new QDoubleSpinBox(dockWidgetContents_3);
        animWeightInput->setObjectName(QStringLiteral("animWeightInput"));
        sizePolicy4.setHeightForWidth(animWeightInput->sizePolicy().hasHeightForWidth());
        animWeightInput->setSizePolicy(sizePolicy4);
        animWeightInput->setMinimumSize(QSize(40, 0));
        animWeightInput->setMaximumSize(QSize(43, 16777215));
        animWeightInput->setFont(font1);
        animWeightInput->setMaximum(10);
        animWeightInput->setSingleStep(0.1);
        animWeightInput->setValue(1);

        horizontalLayout_6->addWidget(animWeightInput);

        animEasingSelect = new QComboBox(dockWidgetContents_3);
        animEasingSelect->setObjectName(QStringLiteral("animEasingSelect"));
        QSizePolicy sizePolicy6(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(animEasingSelect->sizePolicy().hasHeightForWidth());
        animEasingSelect->setSizePolicy(sizePolicy6);

        horizontalLayout_6->addWidget(animEasingSelect);

        animLoopingSelect = new QComboBox(dockWidgetContents_3);
        animLoopingSelect->setObjectName(QStringLiteral("animLoopingSelect"));
        sizePolicy6.setHeightForWidth(animLoopingSelect->sizePolicy().hasHeightForWidth());
        animLoopingSelect->setSizePolicy(sizePolicy6);

        horizontalLayout_6->addWidget(animLoopingSelect);


        verticalLayout_5->addLayout(horizontalLayout_6);


        horizontalLayout_4->addLayout(verticalLayout_5);

        horizontalSpacer = new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);


        verticalLayout_3->addLayout(horizontalLayout_4);

        dockAnimation->setWidget(dockWidgetContents_3);
        qtMainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockAnimation);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuRenderer->menuAction());
        menuBar->addAction(menuInfos->menuAction());
        menuBar->addAction(menuCamera->menuAction());
        menuBar->addAction(menuRender_Flags->menuAction());
        menuBar->addAction(menuAnimation->menuAction());
        menuBar->addAction(menuRay_Tracing->menuAction());
        menuBar->addAction(menuPath_tracing->menuAction());
        menuBar->addAction(menuWindow->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuFile->addAction(menuLoad_Scene->menuAction());
        menuFile->addSeparator();
        menuFile->addAction(action_Quit);
        menuLoad_Scene->addAction(menuGeneral_Scenes->menuAction());
        menuLoad_Scene->addAction(menuShader_Scenes->menuAction());
        menuLoad_Scene->addAction(menuAnimation_Scenes->menuAction());
        menuLoad_Scene->addAction(menuRay_Tracing_Scenes->menuAction());
        menuGeneral_Scenes->addAction(actionSmall_Test_Scene);
        menuGeneral_Scenes->addAction(actionLarge_Model);
        menuGeneral_Scenes->addAction(actionFigure);
        menuGeneral_Scenes->addAction(actionMesh_Loader);
        menuGeneral_Scenes->addAction(actionTexture_Blending);
        menuGeneral_Scenes->addAction(actionTexture_Filtering);
        menuGeneral_Scenes->addAction(actionFrustum_Culling_1);
        menuGeneral_Scenes->addAction(actionFrustum_Culling_2);
        menuShader_Scenes->addAction(actionPer_Vertex_Lighting);
        menuShader_Scenes->addAction(actionPer_Pixel_Lighting);
        menuShader_Scenes->addAction(actionPer_Vertex_Wave);
        menuShader_Scenes->addAction(actionWater);
        menuShader_Scenes->addAction(actionBump_Mapping);
        menuShader_Scenes->addAction(actionParallax_Mapping);
        menuShader_Scenes->addAction(actionGlass_Shader);
        menuShader_Scenes->addAction(actionEarth_Shader);
        menuRay_Tracing_Scenes->addAction(actionRT_Spheres);
        menuRay_Tracing_Scenes->addAction(actionRT_Muttenzer_Box);
        menuRay_Tracing_Scenes->addAction(actionRT_Depth_of_Field);
        menuRay_Tracing_Scenes->addAction(actionRT_Soft_Shadows);
        menuRay_Tracing_Scenes->addAction(actionRT_Lens);
        menuAnimation_Scenes->addAction(actionMass_Animation);
        menuAnimation_Scenes->addAction(actionSkeletal_Animation);
        menuAnimation_Scenes->addAction(actionAstroboy_Army_GPU);
        menuAnimation_Scenes->addAction(actionAstroboy_Army_CPU);
        menuCamera->addAction(actionReset);
        menuCamera->addAction(actionUse_SceneView_Camera);
        menuCamera->addAction(menuProjection->menuAction());
        menuCamera->addAction(menuCamAnimation->menuAction());
        menuCamera->addAction(menuField_of_View->menuAction());
        menuCamera->addAction(menuForward_Speed->menuAction());
        menuProjection->addAction(actionPerspective);
        menuProjection->addAction(actionOrthographic);
        menuProjection->addAction(menuStereo->menuAction());
        menuStereo->addAction(actionSide_by_side);
        menuStereo->addAction(actionSide_by_side_proportional);
        menuStereo->addAction(actionSide_by_side_distorted);
        menuStereo->addAction(actionLine_by_line);
        menuStereo->addAction(actionColumn_by_column);
        menuStereo->addAction(actionPixel_by_pixel);
        menuStereo->addAction(actionColor_Red_Cyan);
        menuStereo->addAction(actionColor_Red_Green);
        menuStereo->addAction(actionColor_Red_Blue);
        menuStereo->addAction(actionColor_Cyan_Yellow);
        menuStereo->addAction(menuEye_Separation->menuAction());
        menuStereo->addAction(menuFocal_Distance->menuAction());
        menuEye_Separation->addAction(action_eyeSepInc10);
        menuEye_Separation->addAction(action_eyeSepDec10);
        menuFocal_Distance->addAction(action_focalDistInc);
        menuFocal_Distance->addAction(action_focalDistDec);
        menuField_of_View->addAction(action_fovInc10);
        menuField_of_View->addAction(action_fovDec10);
        menuCamAnimation->addAction(actionTurntable_Y_up);
        menuCamAnimation->addAction(actionTurntable_Z_up);
        menuCamAnimation->addAction(actionWalking_Y_up);
        menuCamAnimation->addAction(actionWalking_Z_up);
        menuForward_Speed->addAction(action_speedInc);
        menuForward_Speed->addAction(action_speedDec);
        menuRender_Flags->addAction(actionAntialiasing);
        menuRender_Flags->addAction(actionView_Frustum_Culling);
        menuRender_Flags->addAction(actionDepthTest);
        menuRender_Flags->addAction(actionSlowdown_on_Idle);
        menuRender_Flags->addSeparator();
        menuRender_Flags->addAction(actionShow_Normals);
        menuRender_Flags->addAction(actionShow_Wired_Mesh);
        menuRender_Flags->addAction(actionShow_Bounding_Boxes);
        menuRender_Flags->addAction(actionShow_Axis);
        menuRender_Flags->addAction(actionShow_Backfaces);
        menuRender_Flags->addAction(actionShow_Voxels);
        menuRender_Flags->addAction(actionTextures_off);
        menuRender_Flags->addAction(actionAnimation_off);
        menuInfos->addAction(actionShow_DockScenegraph);
        menuInfos->addAction(actionShow_DockProperties);
        menuInfos->addAction(actionShow_Animation_Controler);
        menuInfos->addAction(actionShow_Toolbar);
        menuInfos->addAction(actionShow_Statusbar);
        menuInfos->addSeparator();
        menuInfos->addAction(actionShow_Statistics);
        menuInfos->addAction(actionShow_Scene_Info);
        menuInfos->addAction(actionShow_Menu);
        menuInfos->addSeparator();
        menuRenderer->addAction(actionOpenGL);
        menuRenderer->addAction(actionRay_Tracer);
        menuRenderer->addAction(actionPath_Tracer);
        menuRay_Tracing->addAction(actionRender_to_depth_1);
        menuRay_Tracing->addAction(actionRender_to_depth_2);
        menuRay_Tracing->addAction(actionRender_to_depth_5);
        menuRay_Tracing->addAction(actionRender_to_max_depth);
        menuRay_Tracing->addAction(actionConstant_Redering);
        menuRay_Tracing->addAction(actionRender_Distributed_RT_features);
        menuPath_tracing->addAction(action1_Sample);
        menuPath_tracing->addAction(action10_Samples);
        menuPath_tracing->addAction(action100_Sample);
        menuPath_tracing->addAction(action1000_Samples);
        menuPath_tracing->addAction(action10000_Samples);
        menuWindow->addAction(actionFullscreen);
        menuWindow->addSeparator();
        menuWindow->addAction(actionSplit_active_view_horizontally);
        menuWindow->addAction(actionSplit_active_view_vertically);
        menuWindow->addAction(actionSplit_into_4_views);
        menuWindow->addAction(actionDelete_active_view);
        menuWindow->addAction(actionSingle_view);
        menuHelp->addAction(actionAbout_SLProject);
        menuHelp->addAction(actionAbout_Qt);
        menuHelp->addAction(actionCredits);
        menuAnimation->addAction(actionAnimPlayForward);
        menuAnimation->addAction(actionAnimPlayBackward);
        menuAnimation->addAction(actionAnimPause);
        menuAnimation->addAction(actionAnimStop);
        menuAnimation->addAction(actionAnimNextKeyframe);
        menuAnimation->addAction(actionAnimPrevKeyframe);
        menuAnimation->addAction(actionAnimSkipToEnd);
        menuAnimation->addAction(actionAnimSkipToStart);
        toolBar->addAction(actionOpenGL);
        toolBar->addAction(actionRay_Tracer);
        toolBar->addSeparator();
        toolBar->addAction(actionPerspective);
        toolBar->addAction(actionOrthographic);
        toolBar->addSeparator();
        toolBar->addAction(actionShow_Wired_Mesh);
        toolBar->addAction(actionShow_Normals);
        toolBar->addAction(actionShow_Bounding_Boxes);
        toolBar->addAction(actionShow_Voxels);
        toolBar->addAction(actionShow_Axis);
        toolBar->addAction(actionShow_Backfaces);
        toolBar->addAction(actionTextures_off);
        toolBar->addSeparator();
        toolBar->addAction(actionAnimation_off);
        toolBar->addSeparator();
        toolBar->addAction(actionSplit_active_view_vertically);
        toolBar->addAction(actionSplit_active_view_horizontally);
        toolBar->addAction(actionSplit_into_4_views);
        toolBar->addAction(actionDelete_active_view);
        toolBar->addAction(actionSingle_view);

        retranslateUi(qtMainWindow);
        QObject::connect(actionAnimPlayForward, SIGNAL(triggered()), animPlayForwardButton, SLOT(click()));
        QObject::connect(actionAnimPlayBackward, SIGNAL(triggered()), animPlayBackwardButton, SLOT(click()));
        QObject::connect(actionAnimPause, SIGNAL(triggered()), animPauseButton, SLOT(click()));
        QObject::connect(actionAnimNextKeyframe, SIGNAL(triggered()), animNextKeyframeButton, SLOT(click()));
        QObject::connect(actionAnimPrevKeyframe, SIGNAL(triggered()), animPrevKeyframeButton, SLOT(click()));
        QObject::connect(actionAnimSkipToEnd, SIGNAL(triggered()), animSkipEndButton, SLOT(click()));
        QObject::connect(actionAnimSkipToStart, SIGNAL(triggered()), animSkipStartButton, SLOT(click()));
        QObject::connect(actionAnimStop, SIGNAL(triggered()), animStopButton, SLOT(click()));

        QMetaObject::connectSlotsByName(qtMainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *qtMainWindow)
    {
        qtMainWindow->setWindowTitle(QApplication::translate("qtMainWindow", "qtMainWindow", 0));
        action_Quit->setText(QApplication::translate("qtMainWindow", "Quit", 0));
        actionSmall_Test_Scene->setText(QApplication::translate("qtMainWindow", "Small Test Scene", 0));
#ifndef QT_NO_STATUSTIP
        actionSmall_Test_Scene->setStatusTip(QApplication::translate("qtMainWindow", "Loads a small predefined scene", 0));
#endif // QT_NO_STATUSTIP
        actionLarge_Model->setText(QApplication::translate("qtMainWindow", "Large Model", 0));
#ifndef QT_NO_STATUSTIP
        actionLarge_Model->setStatusTip(QApplication::translate("qtMainWindow", "Loads a large model xyzrgb_dragon.ply (137 MB) form https://graphics.stanford.edu/data/3Dscanrep", 0));
#endif // QT_NO_STATUSTIP
        actionFigure->setText(QApplication::translate("qtMainWindow", "Figure", 0));
#ifndef QT_NO_STATUSTIP
        actionFigure->setStatusTip(QApplication::translate("qtMainWindow", "Loads a hierarchical figure with some animated objects ", 0));
#endif // QT_NO_STATUSTIP
        actionMesh_Loader->setText(QApplication::translate("qtMainWindow", "Mesh Loader", 0));
#ifndef QT_NO_STATUSTIP
        actionMesh_Loader->setStatusTip(QApplication::translate("qtMainWindow", "Loads some models from different mesh formats", 0));
#endif // QT_NO_STATUSTIP
        actionTexture_Blending->setText(QApplication::translate("qtMainWindow", "Texture Blending", 0));
#ifndef QT_NO_STATUSTIP
        actionTexture_Blending->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with transparent trees for depth sort testing", 0));
#endif // QT_NO_STATUSTIP
        actionFrustum_Culling_1->setText(QApplication::translate("qtMainWindow", "Frustum Culling 1", 0));
#ifndef QT_NO_STATUSTIP
        actionFrustum_Culling_1->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with many spheres out of the view frustum for testing the frustum culling.", 0));
#endif // QT_NO_STATUSTIP
        actionFrustum_Culling_2->setText(QApplication::translate("qtMainWindow", "Frustum Culling 2", 0));
#ifndef QT_NO_STATUSTIP
        actionFrustum_Culling_2->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene many animated figures for testing the update traversal", 0));
#endif // QT_NO_STATUSTIP
        actionTexture_Filtering->setText(QApplication::translate("qtMainWindow", "Texture Filtering", 0));
#ifndef QT_NO_STATUSTIP
        actionTexture_Filtering->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene that tests all texture filtering modes", 0));
#endif // QT_NO_STATUSTIP
        actionPer_Vertex_Lighting->setText(QApplication::translate("qtMainWindow", "Per Vertex Lighting", 0));
#ifndef QT_NO_STATUSTIP
        actionPer_Vertex_Lighting->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene for standard per vertex lighting", 0));
#endif // QT_NO_STATUSTIP
        actionPer_Pixel_Lighting->setText(QApplication::translate("qtMainWindow", "Per Pixel Lighting", 0));
#ifndef QT_NO_STATUSTIP
        actionPer_Pixel_Lighting->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene for standard per pixel lighting", 0));
#endif // QT_NO_STATUSTIP
        actionPer_Vertex_Wave->setText(QApplication::translate("qtMainWindow", "Per Vertex Wave", 0));
#ifndef QT_NO_STATUSTIP
        actionPer_Vertex_Wave->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with a vertex wave shader", 0));
#endif // QT_NO_STATUSTIP
        actionWater->setText(QApplication::translate("qtMainWindow", "Water Shader", 0));
#ifndef QT_NO_STATUSTIP
        actionWater->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with a water shader with reflections and refractions", 0));
#endif // QT_NO_STATUSTIP
        actionBump_Mapping->setText(QApplication::translate("qtMainWindow", "Bump Mapping", 0));
#ifndef QT_NO_STATUSTIP
        actionBump_Mapping->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with texture bump mapping", 0));
#endif // QT_NO_STATUSTIP
        actionParallax_Mapping->setText(QApplication::translate("qtMainWindow", "Parallax Mapping", 0));
#ifndef QT_NO_STATUSTIP
        actionParallax_Mapping->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with texture parallax mapping", 0));
#endif // QT_NO_STATUSTIP
        actionGlass_Shader->setText(QApplication::translate("qtMainWindow", "Glass Shader", 0));
#ifndef QT_NO_STATUSTIP
        actionGlass_Shader->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with various revolving objects", 0));
#endif // QT_NO_STATUSTIP
        actionEarth_Shader->setText(QApplication::translate("qtMainWindow", "Earth Shader", 0));
#ifndef QT_NO_STATUSTIP
        actionEarth_Shader->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with a special earth shader using 6 different textures", 0));
#endif // QT_NO_STATUSTIP
        actionRT_Spheres->setText(QApplication::translate("qtMainWindow", "Spheres", 0));
#ifndef QT_NO_STATUSTIP
        actionRT_Spheres->setStatusTip(QApplication::translate("qtMainWindow", "Loads a classic scene for ray tracing.", 0));
#endif // QT_NO_STATUSTIP
        actionRT_Muttenzer_Box->setText(QApplication::translate("qtMainWindow", "Muttenzer Box", 0));
#ifndef QT_NO_STATUSTIP
        actionRT_Muttenzer_Box->setStatusTip(QApplication::translate("qtMainWindow", "Loads a classic scene for ray tracing or path tracing", 0));
#endif // QT_NO_STATUSTIP
        actionRT_Depth_of_Field->setText(QApplication::translate("qtMainWindow", "Depth of Field", 0));
#ifndef QT_NO_STATUSTIP
        actionRT_Depth_of_Field->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene for testing the depth of field feature in distributed ray tracing", 0));
#endif // QT_NO_STATUSTIP
        actionRT_Soft_Shadows->setText(QApplication::translate("qtMainWindow", "Soft Shadows", 0));
#ifndef QT_NO_STATUSTIP
        actionRT_Soft_Shadows->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene for testing the smooth shadow feature from distributed ray tracing", 0));
#endif // QT_NO_STATUSTIP
        actionReset->setText(QApplication::translate("qtMainWindow", "Reset", 0));
#ifndef QT_NO_STATUSTIP
        actionReset->setStatusTip(QApplication::translate("qtMainWindow", "Resets the camera position and projection. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionPerspective->setText(QApplication::translate("qtMainWindow", "Perspective", 0));
#ifndef QT_NO_TOOLTIP
        actionPerspective->setToolTip(QApplication::translate("qtMainWindow", "Perspective projection", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        actionPerspective->setStatusTip(QApplication::translate("qtMainWindow", " Shows the current view in perspective projection. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionOrthographic->setText(QApplication::translate("qtMainWindow", "Orthographic", 0));
#ifndef QT_NO_TOOLTIP
        actionOrthographic->setToolTip(QApplication::translate("qtMainWindow", "Orthographic projection", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        actionOrthographic->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in orthographic projection without perspective shortening.", 0));
#endif // QT_NO_STATUSTIP
        actionSide_by_side->setText(QApplication::translate("qtMainWindow", "Side by side ", 0));
#ifndef QT_NO_STATUSTIP
        actionSide_by_side->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in side by side stereo projection. This type is mostly used by HDMI tv's.", 0));
#endif // QT_NO_STATUSTIP
        actionSide_by_side_proportional->setText(QApplication::translate("qtMainWindow", "Side by side proportional", 0));
#ifndef QT_NO_STATUSTIP
        actionSide_by_side_proportional->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in proportional side by side stereo projection.", 0));
#endif // QT_NO_STATUSTIP
        actionSide_by_side_distorted->setText(QApplication::translate("qtMainWindow", "Side by side distorted", 0));
#ifndef QT_NO_STATUSTIP
        actionSide_by_side_distorted->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in side by distorted side stereo projection. This type is used by the Oculus Rift.", 0));
#endif // QT_NO_STATUSTIP
        actionLine_by_line->setText(QApplication::translate("qtMainWindow", "Line by line", 0));
#ifndef QT_NO_STATUSTIP
        actionLine_by_line->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in line by line stereo projection used by some tv's.", 0));
#endif // QT_NO_STATUSTIP
        actionColumn_by_column->setText(QApplication::translate("qtMainWindow", "Column by column", 0));
#ifndef QT_NO_STATUSTIP
        actionColumn_by_column->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in columns by column stereo projection used by some tv.", 0));
#endif // QT_NO_STATUSTIP
        actionPixel_by_pixel->setText(QApplication::translate("qtMainWindow", "Pixel by pixel", 0));
#ifndef QT_NO_STATUSTIP
        actionPixel_by_pixel->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in chessboard pattern stereo projection used by some tv's.", 0));
#endif // QT_NO_STATUSTIP
        actionColor_Red_Cyan->setText(QApplication::translate("qtMainWindow", "Color Red-Cyan", 0));
#ifndef QT_NO_STATUSTIP
        actionColor_Red_Cyan->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in anaglyth stereo projections with the colors red and cyan.", 0));
#endif // QT_NO_STATUSTIP
        actionColor_Red_Green->setText(QApplication::translate("qtMainWindow", "Color Red-Green", 0));
#ifndef QT_NO_STATUSTIP
        actionColor_Red_Green->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in anaglyth stereo projections with the colors red and green.", 0));
#endif // QT_NO_STATUSTIP
        actionColor_Red_Blue->setText(QApplication::translate("qtMainWindow", "Color Red-Blue", 0));
#ifndef QT_NO_STATUSTIP
        actionColor_Red_Blue->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in anaglyth stereo projections with the colors red and blue.", 0));
#endif // QT_NO_STATUSTIP
        actionColor_Cyan_Yellow->setText(QApplication::translate("qtMainWindow", "Color Yellow-Blue", 0));
#ifndef QT_NO_STATUSTIP
        actionColor_Cyan_Yellow->setStatusTip(QApplication::translate("qtMainWindow", "Shows the current view in anaglyth stereo projections with the colors cyan and yellow.", 0));
#endif // QT_NO_STATUSTIP
        action_eyeSepInc10->setText(QApplication::translate("qtMainWindow", "+ 10%", 0));
#ifndef QT_NO_STATUSTIP
        action_eyeSepInc10->setStatusTip(QApplication::translate("qtMainWindow", "Increases the eye separation distance by 10%", 0));
#endif // QT_NO_STATUSTIP
        action_eyeSepDec10->setText(QApplication::translate("qtMainWindow", "- 10%", 0));
#ifndef QT_NO_STATUSTIP
        action_eyeSepDec10->setStatusTip(QApplication::translate("qtMainWindow", "Decreases the eye separation distance by 10%", 0));
#endif // QT_NO_STATUSTIP
        action_focalDistInc->setText(QApplication::translate("qtMainWindow", "+ 20%", 0));
#ifndef QT_NO_STATUSTIP
        action_focalDistInc->setStatusTip(QApplication::translate("qtMainWindow", "Increases the focal distance to the zero parallax plan by 20%", 0));
#endif // QT_NO_STATUSTIP
        action_focalDistDec->setText(QApplication::translate("qtMainWindow", "- 20%", 0));
#ifndef QT_NO_STATUSTIP
        action_focalDistDec->setStatusTip(QApplication::translate("qtMainWindow", "Decreases the focal distance to the zero parallax plan by 20%", 0));
#endif // QT_NO_STATUSTIP
        action_fovInc10->setText(QApplication::translate("qtMainWindow", "+ 10%", 0));
        action_fovDec10->setText(QApplication::translate("qtMainWindow", "- 10%", 0));
        actionTurntable_Y_up->setText(QApplication::translate("qtMainWindow", "Turntable Y up", 0));
        actionTurntable_Y_up->setIconText(QApplication::translate("qtMainWindow", "Turntable Rotation Y-axis up", 0));
#ifndef QT_NO_STATUSTIP
        actionTurntable_Y_up->setStatusTip(QApplication::translate("qtMainWindow", "Sets the current mouse camera animation to a turntable rotation around the y-axis.", 0));
#endif // QT_NO_STATUSTIP
        actionTurntable_Z_up->setText(QApplication::translate("qtMainWindow", "Turntable Z up", 0));
#ifndef QT_NO_STATUSTIP
        actionTurntable_Z_up->setStatusTip(QApplication::translate("qtMainWindow", "Sets the current mouse camera animation to a turntable rotation around the z-axis.", 0));
#endif // QT_NO_STATUSTIP
        actionWalking_Y_up->setText(QApplication::translate("qtMainWindow", "Walking Y up", 0));
#ifndef QT_NO_STATUSTIP
        actionWalking_Y_up->setStatusTip(QApplication::translate("qtMainWindow", "Sets the current mouse camera animation to a walking motion with head rotation around the y-axis.", 0));
#endif // QT_NO_STATUSTIP
        actionWalking_Z_up->setText(QApplication::translate("qtMainWindow", "Walking Z up", 0));
#ifndef QT_NO_STATUSTIP
        actionWalking_Z_up->setStatusTip(QApplication::translate("qtMainWindow", "Sets the current mouse camera animation to a walking motion with head rotation around the z-axis.", 0));
#endif // QT_NO_STATUSTIP
        action_speedInc->setText(QApplication::translate("qtMainWindow", "+ 20%", 0));
#ifndef QT_NO_STATUSTIP
        action_speedInc->setStatusTip(QApplication::translate("qtMainWindow", "Increases the walking motion speed by 20%", 0));
#endif // QT_NO_STATUSTIP
        action_speedDec->setText(QApplication::translate("qtMainWindow", "- 20%", 0));
#ifndef QT_NO_STATUSTIP
        action_speedDec->setStatusTip(QApplication::translate("qtMainWindow", "Decreases the walking motion speed by 20%", 0));
#endif // QT_NO_STATUSTIP
        actionAntialiasing->setText(QApplication::translate("qtMainWindow", "Do Antialiasing", 0));
#ifndef QT_NO_STATUSTIP
        actionAntialiasing->setStatusTip(QApplication::translate("qtMainWindow", "Turns on the antialiasing for smooth edges if the graphic acceleration is capable for it.", 0));
#endif // QT_NO_STATUSTIP
        actionView_Frustum_Culling->setText(QApplication::translate("qtMainWindow", "Do View Frustum Culling", 0));
#ifndef QT_NO_STATUSTIP
        actionView_Frustum_Culling->setStatusTip(QApplication::translate("qtMainWindow", "Turns on the frustum culling optimization, so that only objects are rendered that are visible.", 0));
#endif // QT_NO_STATUSTIP
        actionSlowdown_on_Idle->setText(QApplication::translate("qtMainWindow", "Slowdown on Idle", 0));
#ifndef QT_NO_STATUSTIP
        actionSlowdown_on_Idle->setStatusTip(QApplication::translate("qtMainWindow", "Slows down the refresh rate if no animation is going on.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Statistics->setText(QApplication::translate("qtMainWindow", "Show Scene Statistics", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Statistics->setStatusTip(QApplication::translate("qtMainWindow", "Shows some rendering and scenegraph statistics in the scene view window", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Scene_Info->setText(QApplication::translate("qtMainWindow", "Show Scene Info", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Scene_Info->setStatusTip(QApplication::translate("qtMainWindow", "Shows some scene specific info in the scene view window", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Menu->setText(QApplication::translate("qtMainWindow", "Show Scene Menu", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Menu->setStatusTip(QApplication::translate("qtMainWindow", "Shows the alternative menu rendered with OpenGL", 0));
#endif // QT_NO_STATUSTIP
        actionAbout_SLProject->setText(QApplication::translate("qtMainWindow", "About SLProject", 0));
#ifndef QT_NO_STATUSTIP
        actionAbout_SLProject->setStatusTip(QApplication::translate("qtMainWindow", "Shows some SLProject information", 0));
#endif // QT_NO_STATUSTIP
        actionCredits->setText(QApplication::translate("qtMainWindow", "About External Libraries", 0));
#ifndef QT_NO_STATUSTIP
        actionCredits->setStatusTip(QApplication::translate("qtMainWindow", "Shows the Credits dialog about the external libraries used in this project.", 0));
#endif // QT_NO_STATUSTIP
        actionAbout_Qt->setText(QApplication::translate("qtMainWindow", "About Qt", 0));
#ifndef QT_NO_STATUSTIP
        actionAbout_Qt->setStatusTip(QApplication::translate("qtMainWindow", "Shows the Qt About dialog", 0));
#endif // QT_NO_STATUSTIP
        actionRay_Tracer->setText(QApplication::translate("qtMainWindow", "Ray Tracer", 0));
#ifndef QT_NO_STATUSTIP
        actionRay_Tracer->setStatusTip(QApplication::translate("qtMainWindow", "Renders the image with distributed ray tracing", 0));
#endif // QT_NO_STATUSTIP
        actionPath_Tracer->setText(QApplication::translate("qtMainWindow", "Path Tracer", 0));
#ifndef QT_NO_STATUSTIP
        actionPath_Tracer->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with path tracing", 0));
#endif // QT_NO_STATUSTIP
        actionOpenGL->setText(QApplication::translate("qtMainWindow", "OpenGL", 0));
#ifndef QT_NO_STATUSTIP
        actionOpenGL->setStatusTip(QApplication::translate("qtMainWindow", "Renders the image with core profile OpenGL", 0));
#endif // QT_NO_STATUSTIP
        actionRender_to_depth_1->setText(QApplication::translate("qtMainWindow", "Render to depth 1", 0));
#ifndef QT_NO_STATUSTIP
        actionRender_to_depth_1->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with ray tracing with max. 1 recursion depth.", 0));
#endif // QT_NO_STATUSTIP
        actionRender_to_depth_2->setText(QApplication::translate("qtMainWindow", "Render to depth 2", 0));
#ifndef QT_NO_STATUSTIP
        actionRender_to_depth_2->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with ray tracing with max. 2 recursion depth.", 0));
#endif // QT_NO_STATUSTIP
        actionRender_to_depth_5->setText(QApplication::translate("qtMainWindow", "Render to depth 5", 0));
#ifndef QT_NO_STATUSTIP
        actionRender_to_depth_5->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with ray tracing with max. 5 recursion depth.", 0));
#endif // QT_NO_STATUSTIP
        actionRender_to_max_depth->setText(QApplication::translate("qtMainWindow", "Render to max. depth", 0));
#ifndef QT_NO_STATUSTIP
        actionRender_to_max_depth->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with ray tracing with max. allowed recursion depth.", 0));
#endif // QT_NO_STATUSTIP
        actionConstant_Redering->setText(QApplication::translate("qtMainWindow", "Constant Redering", 0));
#ifndef QT_NO_STATUSTIP
        actionConstant_Redering->setStatusTip(QApplication::translate("qtMainWindow", "Turns on constant ray tracing without antialiasing.", 0));
#endif // QT_NO_STATUSTIP
        actionRender_Distributed_RT_features->setText(QApplication::translate("qtMainWindow", "Render Distributed RT features", 0));
#ifndef QT_NO_STATUSTIP
        actionRender_Distributed_RT_features->setStatusTip(QApplication::translate("qtMainWindow", "Renders with distributed features of ray tracing. If turned off classic Whitted style RT is used.", 0));
#endif // QT_NO_STATUSTIP
        action1_Sample->setText(QApplication::translate("qtMainWindow", "1 Sample", 0));
#ifndef QT_NO_STATUSTIP
        action1_Sample->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with 1 path sample per pixel.", 0));
#endif // QT_NO_STATUSTIP
        action10_Samples->setText(QApplication::translate("qtMainWindow", "10 Samples", 0));
#ifndef QT_NO_STATUSTIP
        action10_Samples->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with 10 path sampled per pixel.", 0));
#endif // QT_NO_STATUSTIP
        action100_Sample->setText(QApplication::translate("qtMainWindow", "100 Samples", 0));
#ifndef QT_NO_STATUSTIP
        action100_Sample->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with 100 path sampled per pixel.", 0));
#endif // QT_NO_STATUSTIP
        action1000_Samples->setText(QApplication::translate("qtMainWindow", "1000 Samples", 0));
#ifndef QT_NO_STATUSTIP
        action1000_Samples->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with 1000 path sampled per pixel.", 0));
#endif // QT_NO_STATUSTIP
        action10000_Samples->setText(QApplication::translate("qtMainWindow", "10000 Samples", 0));
#ifndef QT_NO_STATUSTIP
        action10000_Samples->setStatusTip(QApplication::translate("qtMainWindow", "Renders the scene with 10000 path sampled per pixel.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_DockScenegraph->setText(QApplication::translate("qtMainWindow", "Show Scenegraph Tree", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_DockScenegraph->setStatusTip(QApplication::translate("qtMainWindow", "Shows the node tree of the current scene also called scenegraph", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Statusbar->setText(QApplication::translate("qtMainWindow", "Show Statusbar", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Statusbar->setStatusTip(QApplication::translate("qtMainWindow", "Shows the this statusbar at the bottom of the window.", 0));
#endif // QT_NO_STATUSTIP
        actionDepthTest->setText(QApplication::translate("qtMainWindow", "Do Depth Test", 0));
#ifndef QT_NO_STATUSTIP
        actionDepthTest->setStatusTip(QApplication::translate("qtMainWindow", "Turns on or off the depth test on the depth buffer for hidden line removal.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Normals->setText(QApplication::translate("qtMainWindow", "Show Normals", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Normals->setStatusTip(QApplication::translate("qtMainWindow", "Shows the vertex normals as a blue line. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Wired_Mesh->setText(QApplication::translate("qtMainWindow", "Show Wired Mesh", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Wired_Mesh->setStatusTip(QApplication::translate("qtMainWindow", "Shows the meshes as lined polygons, so that you can see through. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Bounding_Boxes->setText(QApplication::translate("qtMainWindow", "Show Bounding Boxes", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Bounding_Boxes->setStatusTip(QApplication::translate("qtMainWindow", "Shows the axis aligned bounding box of the node. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Axis->setText(QApplication::translate("qtMainWindow", "Show Axis", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Axis->setStatusTip(QApplication::translate("qtMainWindow", "Shows the central orientation axis of the node. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Backfaces->setText(QApplication::translate("qtMainWindow", "Show Backfaces", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Backfaces->setStatusTip(QApplication::translate("qtMainWindow", "Turns off the default backface culling. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Voxels->setText(QApplication::translate("qtMainWindow", "Show Voxels", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_Voxels->setStatusTip(QApplication::translate("qtMainWindow", "Shows the voxels of the uniform grid that contain triangles in cyan color. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionTextures_off->setText(QApplication::translate("qtMainWindow", "Textures off", 0));
#ifndef QT_NO_STATUSTIP
        actionTextures_off->setStatusTip(QApplication::translate("qtMainWindow", "Disables the texture rendering. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionAnimation_off->setText(QApplication::translate("qtMainWindow", "Animation off", 0));
#ifndef QT_NO_STATUSTIP
        actionAnimation_off->setStatusTip(QApplication::translate("qtMainWindow", "Turns off scene wide all animations.", 0));
#endif // QT_NO_STATUSTIP
        actionFullscreen->setText(QApplication::translate("qtMainWindow", "Fullscreen", 0));
#ifndef QT_NO_STATUSTIP
        actionFullscreen->setStatusTip(QApplication::translate("qtMainWindow", "Sets the window to fullscrene mode without menu, tool and statusbar. Use the ESC to end the fullscreen mode.", 0));
#endif // QT_NO_STATUSTIP
        actionShow_DockProperties->setText(QApplication::translate("qtMainWindow", "Show Properties Tree", 0));
#ifndef QT_NO_STATUSTIP
        actionShow_DockProperties->setStatusTip(QApplication::translate("qtMainWindow", "Show the properties of the selected scene node or mesh", 0));
#endif // QT_NO_STATUSTIP
        actionShow_Toolbar->setText(QApplication::translate("qtMainWindow", "Show Toolbar", 0));
        actionSplit_active_view_horizontally->setText(QApplication::translate("qtMainWindow", "Split active view horizontally", 0));
#ifndef QT_NO_STATUSTIP
        actionSplit_active_view_horizontally->setStatusTip(QApplication::translate("qtMainWindow", "Splits the active view into a left and a right view", 0));
#endif // QT_NO_STATUSTIP
        actionSplit_active_view_vertically->setText(QApplication::translate("qtMainWindow", "Split active view vertically", 0));
#ifndef QT_NO_STATUSTIP
        actionSplit_active_view_vertically->setStatusTip(QApplication::translate("qtMainWindow", "Splits the active view into an uppter and a lower view", 0));
#endif // QT_NO_STATUSTIP
        actionDelete_active_view->setText(QApplication::translate("qtMainWindow", "Delete active view", 0));
#ifndef QT_NO_STATUSTIP
        actionDelete_active_view->setStatusTip(QApplication::translate("qtMainWindow", "Deletes the active view if it is not the last one.", 0));
#endif // QT_NO_STATUSTIP
        actionSplit_into_4_views->setText(QApplication::translate("qtMainWindow", "Split into 4 views", 0));
#ifndef QT_NO_TOOLTIP
        actionSplit_into_4_views->setToolTip(QApplication::translate("qtMainWindow", "Split active view into 4 views", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        actionSplit_into_4_views->setStatusTip(QApplication::translate("qtMainWindow", "Splits the active view in 4 equally sized views", 0));
#endif // QT_NO_STATUSTIP
        actionSingle_view->setText(QApplication::translate("qtMainWindow", "Single view", 0));
#ifndef QT_NO_TOOLTIP
        actionSingle_view->setToolTip(QApplication::translate("qtMainWindow", "Single view (keep acive one)", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        actionSingle_view->setStatusTip(QApplication::translate("qtMainWindow", "Deletes all views except the acitve one", 0));
#endif // QT_NO_STATUSTIP
        actionMass_Animation->setText(QApplication::translate("qtMainWindow", "Mass Animation", 0));
#ifndef QT_NO_STATUSTIP
        actionMass_Animation->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with a high number of animations active.", 0));
#endif // QT_NO_STATUSTIP
        actionUse_SceneView_Camera->setText(QApplication::translate("qtMainWindow", "Use SceneView Camera", 0));
#ifndef QT_NO_STATUSTIP
        actionUse_SceneView_Camera->setStatusTip(QApplication::translate("qtMainWindow", "Uses the scene view camera that is not in the scene. Press shift to apply on all views.", 0));
#endif // QT_NO_STATUSTIP
        actionAnimPlayForward->setText(QApplication::translate("qtMainWindow", "Play forward", 0));
#ifndef QT_NO_TOOLTIP
        actionAnimPlayForward->setToolTip(QApplication::translate("qtMainWindow", "Play animation forward", 0));
#endif // QT_NO_TOOLTIP
        actionAnimPause->setText(QApplication::translate("qtMainWindow", "Pause", 0));
        actionAnimPlayBackward->setText(QApplication::translate("qtMainWindow", "Play backward", 0));
        actionAnimNextKeyframe->setText(QApplication::translate("qtMainWindow", "Next keyframe", 0));
        actionAnimPrevKeyframe->setText(QApplication::translate("qtMainWindow", "Previous keyframe", 0));
        actionAnimSkipToStart->setText(QApplication::translate("qtMainWindow", "Skip to first frame", 0));
        actionAnimSkipToEnd->setText(QApplication::translate("qtMainWindow", "Skip to last frame", 0));
        actionAnimStop->setText(QApplication::translate("qtMainWindow", "Stop", 0));
        actionShow_Animation_Controler->setText(QApplication::translate("qtMainWindow", "Show Animation Controller", 0));
        actionSkeletal_Animation->setText(QApplication::translate("qtMainWindow", "Skeletal Animation", 0));
#ifndef QT_NO_STATUSTIP
        actionSkeletal_Animation->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with a few skeletal animation samples.", 0));
#endif // QT_NO_STATUSTIP
        actionAstroboy_Army_GPU->setText(QApplication::translate("qtMainWindow", "Astroboy Army (GPU)", 0));
#ifndef QT_NO_STATUSTIP
        actionAstroboy_Army_GPU->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with many identical animated characters.", 0));
#endif // QT_NO_STATUSTIP
        actionRT_Lens->setText(QApplication::translate("qtMainWindow", "Lens", 0));
        actionAstroboy_Army_CPU->setText(QApplication::translate("qtMainWindow", "Astroboy Army (CPU)", 0));
#ifndef QT_NO_STATUSTIP
        actionAstroboy_Army_CPU->setStatusTip(QApplication::translate("qtMainWindow", "Loads a scene with many identical animated characters.", 0));
#endif // QT_NO_STATUSTIP
        actionSuperAnimation->setText(QApplication::translate("qtMainWindow", "SuperAnimation", 0));
        menuFile->setTitle(QApplication::translate("qtMainWindow", "File", 0));
        menuLoad_Scene->setTitle(QApplication::translate("qtMainWindow", "Load Scene", 0));
        menuGeneral_Scenes->setTitle(QApplication::translate("qtMainWindow", "General Scenes", 0));
        menuShader_Scenes->setTitle(QApplication::translate("qtMainWindow", "Shader Scenes", 0));
        menuRay_Tracing_Scenes->setTitle(QApplication::translate("qtMainWindow", "Ray Tracing Scenes", 0));
        menuAnimation_Scenes->setTitle(QApplication::translate("qtMainWindow", "Animation Scenes", 0));
        menuCamera->setTitle(QApplication::translate("qtMainWindow", "Camera", 0));
        menuProjection->setTitle(QApplication::translate("qtMainWindow", "Projection", 0));
        menuStereo->setTitle(QApplication::translate("qtMainWindow", "Stereo", 0));
        menuEye_Separation->setTitle(QApplication::translate("qtMainWindow", "Eye Separation", 0));
        menuFocal_Distance->setTitle(QApplication::translate("qtMainWindow", "Focal Distance", 0));
        menuField_of_View->setTitle(QApplication::translate("qtMainWindow", "Field of View", 0));
        menuCamAnimation->setTitle(QApplication::translate("qtMainWindow", "Animation", 0));
        menuForward_Speed->setTitle(QApplication::translate("qtMainWindow", "Forward Speed", 0));
        menuRender_Flags->setTitle(QApplication::translate("qtMainWindow", "Renderflags", 0));
        menuInfos->setTitle(QApplication::translate("qtMainWindow", "Infos", 0));
        menuRenderer->setTitle(QApplication::translate("qtMainWindow", "Renderer", 0));
        menuRay_Tracing->setTitle(QApplication::translate("qtMainWindow", "Ray tracing", 0));
        menuPath_tracing->setTitle(QApplication::translate("qtMainWindow", "Path tracing", 0));
        menuWindow->setTitle(QApplication::translate("qtMainWindow", "Window", 0));
        menuHelp->setTitle(QApplication::translate("qtMainWindow", "Help", 0));
        menuAnimation->setTitle(QApplication::translate("qtMainWindow", "Animation", 0));
        dockScenegraph->setWindowTitle(QApplication::translate("qtMainWindow", "Scenegraph", 0));
        toolBar->setWindowTitle(QApplication::translate("qtMainWindow", "toolBar", 0));
        dockProperties->setWindowTitle(QApplication::translate("qtMainWindow", "Properties", 0));
        QTreeWidgetItem *___qtreewidgetitem = propertyTree->headerItem();
        ___qtreewidgetitem->setText(1, QApplication::translate("qtMainWindow", "Value", 0));
        ___qtreewidgetitem->setText(0, QApplication::translate("qtMainWindow", "Property", 0));
        dockAnimation->setWindowTitle(QApplication::translate("qtMainWindow", "Animation", 0));
        animCurrentTimeLabel->setText(QApplication::translate("qtMainWindow", "--:--", 0));
#ifndef QT_NO_TOOLTIP
        animTimelineSlider->setToolTip(QString());
#endif // QT_NO_TOOLTIP
        animDurationLabel->setText(QApplication::translate("qtMainWindow", "--:--", 0));
        animAnimatedObjectSelect->clear();
        animAnimatedObjectSelect->insertItems(0, QStringList()
         << QApplication::translate("qtMainWindow", "Animated objects", 0)
        );
#ifndef QT_NO_TOOLTIP
        animAnimatedObjectSelect->setToolTip(QApplication::translate("qtMainWindow", "Select animation target", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animAnimatedObjectSelect->setStatusTip(QApplication::translate("qtMainWindow", "Select animation target", 0));
#endif // QT_NO_STATUSTIP
        animAnimationSelect->clear();
        animAnimationSelect->insertItems(0, QStringList()
         << QApplication::translate("qtMainWindow", "Animations", 0)
        );
#ifndef QT_NO_TOOLTIP
        animAnimationSelect->setToolTip(QApplication::translate("qtMainWindow", "Select animations", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animAnimationSelect->setStatusTip(QApplication::translate("qtMainWindow", "Select animation of the selected object", 0));
#endif // QT_NO_STATUSTIP
#ifndef QT_NO_TOOLTIP
        animSkipStartButton->setToolTip(QApplication::translate("qtMainWindow", "Skip to start", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animSkipStartButton->setStatusTip(QApplication::translate("qtMainWindow", "Skip to beginning of the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animSkipStartButton->setText(QApplication::translate("qtMainWindow", "skipStart", 0));
#ifndef QT_NO_TOOLTIP
        animPrevKeyframeButton->setToolTip(QApplication::translate("qtMainWindow", "Previous keyframe", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animPrevKeyframeButton->setStatusTip(QApplication::translate("qtMainWindow", "Skip to previous keyframe of the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animPrevKeyframeButton->setText(QApplication::translate("qtMainWindow", "prevKeyframe", 0));
#ifndef QT_NO_TOOLTIP
        animPlayBackwardButton->setToolTip(QApplication::translate("qtMainWindow", "Play backward", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animPlayBackwardButton->setStatusTip(QApplication::translate("qtMainWindow", "Play the selected animation backward", 0));
#endif // QT_NO_STATUSTIP
        animPlayBackwardButton->setText(QApplication::translate("qtMainWindow", "playBackward", 0));
#ifndef QT_NO_TOOLTIP
        animStopButton->setToolTip(QApplication::translate("qtMainWindow", "Disable animation", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animStopButton->setStatusTip(QApplication::translate("qtMainWindow", "Disable the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animStopButton->setText(QApplication::translate("qtMainWindow", "...", 0));
#ifndef QT_NO_TOOLTIP
        animPauseButton->setToolTip(QApplication::translate("qtMainWindow", "Pause", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animPauseButton->setStatusTip(QApplication::translate("qtMainWindow", "Pause the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animPauseButton->setText(QApplication::translate("qtMainWindow", "pause", 0));
#ifndef QT_NO_TOOLTIP
        animPlayForwardButton->setToolTip(QApplication::translate("qtMainWindow", "Play forward", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animPlayForwardButton->setStatusTip(QApplication::translate("qtMainWindow", "Play the selected animation forward", 0));
#endif // QT_NO_STATUSTIP
        animPlayForwardButton->setText(QApplication::translate("qtMainWindow", "playForward", 0));
#ifndef QT_NO_TOOLTIP
        animNextKeyframeButton->setToolTip(QApplication::translate("qtMainWindow", "Next keyframe", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animNextKeyframeButton->setStatusTip(QApplication::translate("qtMainWindow", "Skip to next keyframe of the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animNextKeyframeButton->setText(QApplication::translate("qtMainWindow", "nextKeyframe", 0));
#ifndef QT_NO_TOOLTIP
        animSkipEndButton->setToolTip(QApplication::translate("qtMainWindow", "Skip to end", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animSkipEndButton->setStatusTip(QApplication::translate("qtMainWindow", "Skip to end of the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animSkipEndButton->setText(QApplication::translate("qtMainWindow", "skipEnd", 0));
#ifndef QT_NO_TOOLTIP
        animSpeedInput->setToolTip(QApplication::translate("qtMainWindow", "Playback speed", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animSpeedInput->setStatusTip(QApplication::translate("qtMainWindow", "Set the playback speed for the selected animation", 0));
#endif // QT_NO_STATUSTIP
#ifndef QT_NO_TOOLTIP
        animWeightInput->setToolTip(QApplication::translate("qtMainWindow", "Animation weight", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animWeightInput->setStatusTip(QApplication::translate("qtMainWindow", "Set the weight for the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animEasingSelect->clear();
        animEasingSelect->insertItems(0, QStringList()
         << QApplication::translate("qtMainWindow", "linear", 0)
         << QApplication::translate("qtMainWindow", "quad in", 0)
         << QApplication::translate("qtMainWindow", "quad out", 0)
         << QApplication::translate("qtMainWindow", "quadc in > out", 0)
         << QApplication::translate("qtMainWindow", "quad out > in", 0)
         << QApplication::translate("qtMainWindow", "cubic in", 0)
         << QApplication::translate("qtMainWindow", "cubic out", 0)
         << QApplication::translate("qtMainWindow", "cubic in > out", 0)
         << QApplication::translate("qtMainWindow", "cubic out > in", 0)
         << QApplication::translate("qtMainWindow", "quartic in", 0)
         << QApplication::translate("qtMainWindow", "quartic out", 0)
         << QApplication::translate("qtMainWindow", "quartic in > out", 0)
         << QApplication::translate("qtMainWindow", "quartic out > in", 0)
         << QApplication::translate("qtMainWindow", "quintic in", 0)
         << QApplication::translate("qtMainWindow", "quintic out", 0)
         << QApplication::translate("qtMainWindow", "quintic in > out", 0)
         << QApplication::translate("qtMainWindow", "quintic out > in", 0)
         << QApplication::translate("qtMainWindow", "sine in", 0)
         << QApplication::translate("qtMainWindow", "sine out", 0)
         << QApplication::translate("qtMainWindow", "sine in > out", 0)
         << QApplication::translate("qtMainWindow", "sine out > in", 0)
        );
#ifndef QT_NO_TOOLTIP
        animEasingSelect->setToolTip(QApplication::translate("qtMainWindow", "Easing curve", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animEasingSelect->setStatusTip(QApplication::translate("qtMainWindow", "Set the easing curve for the selected animation", 0));
#endif // QT_NO_STATUSTIP
        animLoopingSelect->clear();
        animLoopingSelect->insertItems(0, QStringList()
         << QApplication::translate("qtMainWindow", "Play once", 0)
         << QApplication::translate("qtMainWindow", "Loop", 0)
         << QApplication::translate("qtMainWindow", "Ping pong", 0)
         << QApplication::translate("qtMainWindow", "Ping pong loop", 0)
        );
#ifndef QT_NO_TOOLTIP
        animLoopingSelect->setToolTip(QApplication::translate("qtMainWindow", "Loop Behaviour", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        animLoopingSelect->setStatusTip(QApplication::translate("qtMainWindow", "Select the looping behaviour for the selected animation", 0));
#endif // QT_NO_STATUSTIP
    } // retranslateUi

};

namespace Ui {
    class qtMainWindow: public Ui_qtMainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTMAINWINDOW_H
