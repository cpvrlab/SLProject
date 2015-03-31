//#############################################################################
//  File:      qtMainWindow.h
//  Purpose:   Main window class declaration
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef QTMAINWINDOW_H
#define QTMAINWINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <SL.h>
#include <SLNode.h>
#include <SLMesh.h>
#include <SLAnimPlayback.h>
#include "qtGLWidget.h"
#include "qtNodeTreeItem.h"
#include "ui_qtMainWindow.h"

namespace Ui {
class qtMainWindow;
}

class qtMainWindow : public QMainWindow
{
	Q_OBJECT
	friend qtGLWidget;

	public:
        explicit qtMainWindow(QWidget *parent, SLVstring cmdLineArgs);
        ~qtMainWindow();

        void        setMenuState();
        void        beforeSceneLoad();
        void        afterSceneLoad();
        void        buildNodeTree();
        void        updateAnimationList();
        void        updateAnimationTimeline();
        void        selectAnimationFromNode(SLNode* node);
        void        buildPropertyTree();
        void        addNodeTreeItem(SLNode* node,
							        QTreeWidget* tree,
							        qtNodeTreeItem* parent);
        void        selectNodeOrMeshItem(SLNode* selectedNode,
                                         SLMesh* selectedMesh);
        void        updateAllGLWidgets();
        void        applyCommandOnSV(const SLCmd cmd);
        qtGLWidget* getOtherGLWidgetInSplitter();

        // Overwritten Event Handlers
        void        resizeEvent(QResizeEvent* event);
        void        changeEvent(QEvent * event);
        void        closeEvent(QCloseEvent *event);

        // Getters
        qtGLWidget* activeGLWidget() {return _activeGLWidget;}
        int         nodeTreeItemCount() {return ui->nodeTree->topLevelItemCount();}

        // Setters
        void        activeGLWidget(qtGLWidget* active) {_activeGLWidget = active;}

	private slots:
        void on_action_Quit_triggered();
        void on_actionSmall_Test_Scene_triggered();
        void on_actionLarge_Model_triggered();
        void on_actionFigure_triggered();
        void on_actionMesh_Loader_triggered();
        void on_actionTexture_Blending_triggered();
        void on_actionTexture_Filtering_triggered();
        void on_actionFrustum_Culling_1_triggered();
        void on_actionFrustum_Culling_2_triggered();

        void on_actionPer_Vertex_Lighting_triggered();
        void on_actionPer_Pixel_Lighting_triggered();
        void on_actionPer_Vertex_Wave_triggered();
        void on_actionWater_triggered();
        void on_actionBump_Mapping_triggered();
        void on_actionParallax_Mapping_triggered();
        void on_actionGlass_Shader_triggered();
        void on_actionEarth_Shader_triggered();

        void on_actionNode_Animation_triggered();
        void on_actionSkeletal_Animation_triggered();
        void on_actionAstroboy_Army_GPU_triggered();
        void on_actionAstroboy_Army_CPU_triggered();
        void on_actionMass_Animation_triggered();

        void on_actionRT_Spheres_triggered();
        void on_actionRT_Muttenzer_Box_triggered();
        void on_actionRT_Soft_Shadows_triggered();
        void on_actionRT_Depth_of_Field_triggered();
        void on_actionRT_Lens_triggered();

        void on_actionReset_triggered();
        void on_actionUse_SceneView_Camera_triggered();
        void on_actionPerspective_triggered();
        void on_actionOrthographic_triggered();
        void on_actionSide_by_side_triggered();
        void on_actionSide_by_side_proportional_triggered();
        void on_actionSide_by_side_distorted_triggered();
        void on_actionLine_by_line_triggered();
        void on_actionColumn_by_column_triggered();
        void on_actionPixel_by_pixel_triggered();
        void on_actionColor_Red_Cyan_triggered();
        void on_actionColor_Red_Green_triggered();
        void on_actionColor_Red_Blue_triggered();
        void on_actionColor_Cyan_Yellow_triggered();
        void on_action_eyeSepInc10_triggered();
        void on_action_eyeSepDec10_triggered();
        void on_action_focalDistInc_triggered();
        void on_action_focalDistDec_triggered();
        void on_action_fovInc10_triggered();
        void on_action_fovDec10_triggered();
        void on_actionTurntable_Y_up_triggered();
        void on_actionTurntable_Z_up_triggered();
        void on_actionWalking_Y_up_triggered();
        void on_actionWalking_Z_up_triggered();
        void on_action_speedInc_triggered();
        void on_action_speedDec_triggered();

        void on_actionAntialiasing_triggered();
        void on_actionView_Frustum_Culling_triggered();
        void on_actionSlowdown_on_Idle_triggered();
        void on_actionDepthTest_triggered();
        void on_actionShow_Statusbar_triggered();
        void on_actionShow_Normals_triggered();
        void on_actionShow_Wired_Mesh_triggered();
        void on_actionShow_Bounding_Boxes_triggered();
        void on_actionShow_Axis_triggered();
        void on_actionShow_Backfaces_triggered();
        void on_actionShow_Voxels_triggered();
        void on_actionTextures_off_triggered();
        void on_actionAnimation_off_triggered();

        void on_actionRay_Tracer_triggered();
        void on_actionOpenGL_triggered();
        void on_actionPath_Tracer_triggered();

        void on_actionShow_DockScenegraph_triggered();
        void on_actionShow_DockProperties_triggered();
        void on_actionShow_Animation_Controler_triggered();
        void on_actionShow_Statistics_triggered();
        void on_actionShow_Scene_Info_triggered();
        void on_actionShow_Menu_triggered();
        void on_actionShow_Toolbar_triggered();
        void on_actionAbout_SLProject_triggered();
        void on_actionCredits_triggered();
        void on_actionAbout_Qt_triggered();

        void on_actionRender_to_depth_1_triggered();
        void on_actionRender_to_depth_2_triggered();
        void on_actionRender_to_depth_5_triggered();
        void on_actionRender_to_max_depth_triggered();
        void on_actionConstant_Redering_triggered();
        void on_actionRender_Distributed_RT_features_triggered();

        void on_action1_Sample_triggered();
        void on_action10_Samples_triggered();
        void on_action100_Sample_triggered();
        void on_action1000_Samples_triggered();
        void on_action10000_Samples_triggered();

        void on_actionFullscreen_triggered();
        void on_actionSplit_active_view_horizontally_triggered();
        void on_actionSplit_active_view_vertically_triggered();
        void on_actionSplit_into_4_views_triggered();
        void on_actionDelete_active_view_triggered();
        void on_actionSingle_view_triggered();

        void on_nodeTree_itemClicked(QTreeWidgetItem *item, int column);
        void on_nodeTree_itemDoubleClicked(QTreeWidgetItem *item, int column);
        void on_propertyTree_itemChanged(QTreeWidgetItem *item, int column);
        void on_dockScenegraph_visibilityChanged(bool visible);
        void on_dockProperties_visibilityChanged(bool visible);
        void on_dockAnimation_visibilityChanged(bool visible);

        // animation
        void on_animAnimatedObjectSelect_currentIndexChanged(int index);
        void on_animAnimationSelect_currentIndexChanged(int index);
        void on_animSkipStartButton_clicked();
        void on_animSkipEndButton_clicked();
        void on_animPrevKeyframeButton_clicked();
        void on_animNextKeyframeButton_clicked();
        void on_animPlayForwardButton_clicked();
        void on_animPlayBackwardButton_clicked();
        void on_animPauseButton_clicked();
        void on_animStopButton_clicked();
        void on_animEasingSelect_currentIndexChanged(int index);
        void on_animLoopingSelect_currentIndexChanged(int index);
        void on_animTimelineSlider_valueChanged(int value);
        void on_animWeightInput_valueChanged(double d);
        void on_animSpeedInput_valueChanged(double d);
        
private:
        Ui::qtMainWindow*  ui;
        std::vector<qtGLWidget*> _allGLWidgets;
        qtGLWidget*       _activeGLWidget;
        qtNodeTreeItem*   _selectedNodeItem;
        QMenu*            _menuFile;
        QMenu*            _menuRenderer;
        QMenu*            _menuInfos;
        QMenu*            _menuCamera;
        QMenu*            _menuAnimation;
        QMenu*            _menuRenderFlags;
        QMenu*            _menuRayTracing;
        QMenu*            _menuPathTracing;
        QMenu*            _menuWindow;
        QMenu*            _menuHelp;

        SLAnimPlayback*   _selectedAnim;
};

#endif // MAINWINDOW_H
