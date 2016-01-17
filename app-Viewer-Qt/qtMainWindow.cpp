//#############################################################################
//  File:      qtMainWindow.cpp
//  Purpose:   Main window class implementation
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "qtMainWindow.h"
#include "ui_qtMainWindow.h"
#include "qtGLWidget.h"
#include "qtAnimationSlider.h"
#include "qtPropertyTreeWidget.h"
#include "qtPropertyTreeItem.h"
#include <qstylefactory.h>
#include <QMessageBox>
#include <QSplitter>
#include <QIcon>
#include <functional>

#include <SL.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLMaterial.h>
#include <SLInterface.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include <SLCamera.h>
#include <SLLight.h>
#include <SLLightRect.h>
#include <SLLightSphere.h>
#include <SLAnimPlayback.h>

using namespace std::placeholders;

// register an sl type to use it as data in combo boxes
Q_DECLARE_METATYPE(SLAnimPlayback*);

//-----------------------------------------------------------------------------
bool qtPropertyTreeWidget::isBeingBuilt = false;
//-----------------------------------------------------------------------------
qtMainWindow::qtMainWindow(QWidget *parent, SLVstring cmdLineArgs) :
   QMainWindow(parent),
   ui(new Ui::qtMainWindow)
{
    ui->setupUi(this);

    _selectedNodeItem = 0;

    _selectedAnim = NULL;

    _menuFile = ui->menuFile;
    _menuCamera = ui->menuCamera;
    _menuAnimation = ui->menuAnimation;
    _menuRenderFlags = ui->menuRender_Flags;
    _menuRenderer = ui->menuRenderer;
    _menuInfos = ui->menuInfos;
    _menuRayTracing = ui->menuRay_Tracing;
    _menuPathTracing = ui->menuPath_tracing;
    _menuWindow = ui->menuWindow;
    _menuHelp = ui->menuHelp;

    qtGLWidget::mainWindow = this;

    // on Mac OSX the sample buffers must be turned on for antialiasing
    QGLFormat format;
    format.defaultFormat();
    format.setSampleBuffers(true);
    format.setProfile(QGLFormat::CoreProfile);
    format.setSwapInterval(1);

    // The composition of widget is as follows:
    //
    // +------------------------------------------------------------------+
    // |  QMainWindow with a QSplitter a central widget                   |
    // |  +-------------------------------------------------------------+ |
    // |  |  QSplitter1                                                 | |
    // |  |  ########################################################## | |
    // |  |  #  QWidget1 for frame                                    # | |
    // |  |  #  +---------------------------------------------------+ # | |
    // |  |  #  |  QGLWidget1 for OpenGL rendering                  | # | |
    // |  |  #  |                                                   | # | |
    // |  |  #  |                                                   | # | |
    // |  |  #  |                                                   | # | |
    // |  |  #  |                                                   | # | |
    // |  |  #  |                                                   | # | |
    // |  |  #  +---------------------------------------------------+ # | |
    // |  |  ########################################################## | |
    // |  +-------------------------------------------------------------+ |
    // +------------------------------------------------------------------+
    //
    // The reason is the possibility for adding new QWidgets to the
    // QSplitter1. See qtGLWidget::split

    // Create splitter & sourrounding frame widget
    QSplitter *splitter  = new QSplitter(Qt::Horizontal);
    QWidget* borderWidget = new QWidget(splitter);
    borderWidget->setLayout(new QHBoxLayout);
    borderWidget->layout()->setMargin(2);
    borderWidget->setStyleSheet("border:2px solid red;");

    // create OpenGL widget. 
    _activeGLWidget = new qtGLWidget(format, borderWidget, "/data/data/ch.fhwn.comgr/files", cmdLineArgs);

    // add glWidget to border and border to splitter and splitter to main window
    borderWidget->layout()->addWidget(_activeGLWidget);
    splitter->addWidget(borderWidget);
    setCentralWidget(splitter);

    splitter->show();
    borderWidget->show();
    _activeGLWidget->show();
}

qtMainWindow::~qtMainWindow()
{
    delete ui;
}

//-----------------------------------------------------------------------------
//! Sets the correct menu checkmarks
void qtMainWindow::setMenuState()
{
    if (!SLScene::current) return;
    SLScene* s = SLScene::current;
    SLSceneView* sv = _activeGLWidget->sv();
    SLCamera* cam = sv->camera();

    // Assemble menu bar
    ui->menuBar->clear();
    ui->menuBar->addMenu(_menuFile);
    ui->menuBar->addMenu(_menuRenderer);
    ui->menuBar->addMenu(_menuInfos);
    ui->menuBar->addMenu(_menuCamera);
    ui->menuBar->addMenu(_menuAnimation);
    ui->menuBar->addMenu(_menuRenderFlags);
    if (sv->renderType()==RT_rt)
        ui->menuBar->addMenu(_menuRayTracing);
    if (sv->renderType()==RT_rt)
        ui->menuBar->addMenu(_menuPathTracing);
    ui->menuBar->addMenu(_menuWindow);
    ui->menuBar->addMenu(_menuHelp);

    // Menu Load Scenes
    ui->actionSmall_Test_Scene->setChecked(s->currentSceneID()==C_sceneMinimal);
    ui->actionLarge_Model->setChecked(s->currentSceneID()==C_sceneLargeModel);
    ui->actionFigure->setChecked(s->currentSceneID()==C_sceneFigure);
    ui->actionMesh_Loader->setChecked(s->currentSceneID()==C_sceneMeshLoad);
    ui->actionTexture_Blending->setChecked(s->currentSceneID()==C_sceneTextureBlend);
    ui->actionTexture_Filtering->setChecked(s->currentSceneID()==C_sceneTextureFilter);
    ui->actionFrustum_Culling_1->setChecked(s->currentSceneID()==C_sceneFrustumCull1);

    ui->actionPer_Vertex_Lighting->setChecked(s->currentSceneID()==C_scenePerVertexBlinn);
    ui->actionPer_Pixel_Lighting->setChecked(s->currentSceneID()==C_scenePerPixelBlinn);
    ui->actionPer_Vertex_Wave->setChecked(s->currentSceneID()==C_scenePerVertexWave);
    ui->actionWater->setChecked(s->currentSceneID()==C_sceneWater);
    ui->actionBump_Mapping->setChecked(s->currentSceneID()==C_sceneBumpNormal);
    ui->actionParallax_Mapping->setChecked(s->currentSceneID()==C_sceneBumpParallax);
    ui->actionGlass_Shader->setChecked(s->currentSceneID()==C_sceneRevolver);
    ui->actionEarth_Shader->setChecked(s->currentSceneID()==C_sceneEarth);

    ui->actionNode_Animation->setChecked(s->currentSceneID()==C_sceneNodeAnimation);
    ui->actionSkeletal_Animation->setChecked(s->currentSceneID()==C_sceneSkeletalAnimation);
    ui->actionAstroboy_Army_CPU->setChecked(s->currentSceneID()==C_sceneAstroboyArmyCPU);
    ui->actionAstroboy_Army_GPU->setChecked(s->currentSceneID()==C_sceneAstroboyArmyGPU);
    ui->actionMass_Animation->setChecked(s->currentSceneID()==C_sceneMassAnimation);

    ui->actionRT_Spheres->setChecked(s->currentSceneID()==C_sceneRTSpheres);
    ui->actionRT_Muttenzer_Box->setChecked(s->currentSceneID()==C_sceneRTMuttenzerBox);
    ui->actionRT_Soft_Shadows->setChecked(s->currentSceneID()==C_sceneRTSoftShadows);
    ui->actionRT_Depth_of_Field->setChecked(s->currentSceneID()==C_sceneRTDoF);
    ui->actionRT_Lens->setChecked(s->currentSceneID()==C_sceneRTLens);

    // Menu Renderer
    ui->actionOpenGL->setChecked(sv->renderType()==RT_gl);
    ui->actionRay_Tracer->setChecked(sv->renderType()==RT_rt);
    ui->actionPath_Tracer->setChecked(sv->renderType()==RT_pt);
    ui->actionRay_Tracer->setEnabled(cam->projection() <= P_monoOrthographic);
    ui->actionPath_Tracer->setEnabled(cam->projection() <= P_monoOrthographic);

    // Menu Info
    ui->actionShow_DockScenegraph->setChecked(ui->dockScenegraph->isVisible());
    ui->actionShow_DockProperties->setChecked(ui->dockProperties->isVisible());
    ui->actionShow_Animation_Controler->setChecked(ui->dockAnimation->isVisible());
    ui->actionShow_Toolbar->setChecked(ui->toolBar->isVisible());
    ui->actionShow_Statusbar->setChecked(ui->statusBar->isVisible());
    ui->actionShow_Statistics->setChecked(sv->showStats());
    ui->actionShow_Scene_Info->setChecked(sv->showInfo());
    ui->actionShow_Menu->setChecked(sv->showMenu());

    // Menu Camera states
    ui->actionUse_SceneView_Camera->setChecked(sv->isSceneViewCameraActive());
    ui->actionPerspective->setChecked(cam->projection() == P_monoPerspective);
    ui->actionOrthographic->setChecked(cam->projection() == P_monoOrthographic);
    ui->menuStereo->setEnabled(sv->renderType()==RT_gl);
    ui->actionSide_by_side->setChecked(cam->projection() == P_stereoSideBySide);
    ui->actionSide_by_side_proportional->setChecked(cam->projection() == P_stereoSideBySideP);
    ui->actionSide_by_side_distorted->setChecked(cam->projection() == P_stereoSideBySideD);
    ui->actionLine_by_line->setChecked(cam->projection() == P_stereoLineByLine);
    ui->actionColumn_by_column->setChecked(cam->projection() == P_stereoColumnByColumn);
    ui->actionPixel_by_pixel->setChecked(cam->projection() == P_stereoPixelByPixel);
    ui->actionColor_Red_Cyan->setChecked(cam->projection() == P_stereoColorRC);
    ui->actionColor_Red_Green->setChecked(cam->projection() == P_stereoColorRG);
    ui->actionColor_Red_Blue->setChecked(cam->projection() == P_stereoColorRB);
    ui->actionColor_Cyan_Yellow->setChecked(cam->projection() == P_stereoColorYB);
    ui->actionTurntable_Y_up->setChecked(cam->camAnim() == CA_turntableYUp);
    ui->actionTurntable_Z_up->setChecked(cam->camAnim() == CA_turntableZUp);
    ui->actionWalking_Y_up->setChecked(cam->camAnim() == CA_walkingYUp);
    ui->actionWalking_Z_up->setChecked(cam->camAnim() == CA_walkingZUp);

    // Menu Render Flags
    ui->actionAntialiasing->setChecked(sv->doMultiSampling());
    ui->actionAntialiasing->setEnabled(sv->hasMultiSampling() || sv->renderType()==RT_rt);
    ui->actionView_Frustum_Culling->setChecked(sv->doFrustumCulling());
    ui->actionSlowdown_on_Idle->setChecked(sv->waitEvents());
    ui->actionDepthTest->setChecked(sv->doDepthTest());
    ui->actionShow_Normals->setChecked(sv->drawBit(SL_DB_NORMALS));
    ui->actionShow_Wired_Mesh->setChecked(sv->drawBit(SL_DB_WIREMESH));
    ui->actionShow_Bounding_Boxes->setChecked(sv->drawBit(SL_DB_BBOX));
    ui->actionShow_Axis->setChecked(sv->drawBit(SL_DB_AXIS));
    ui->actionShow_Voxels->setChecked(sv->drawBit(SL_DB_VOXELS));
    ui->actionShow_Backfaces->setChecked(sv->drawBit(SL_DB_CULLOFF));
    ui->actionTextures_off->setChecked(sv->drawBit(SL_DB_TEXOFF));

    ui->actionAnimation_off->setChecked(s->stopAnimations());

    ui->actionView_Frustum_Culling->setEnabled(sv->renderType()==RT_gl);
    ui->actionSlowdown_on_Idle->setEnabled(sv->renderType()==RT_gl);
    ui->actionDepthTest->setEnabled(sv->renderType()==RT_gl);
    ui->actionShow_Normals->setEnabled(sv->renderType()==RT_gl);
    ui->actionShow_Wired_Mesh->setEnabled(sv->renderType()==RT_gl);
    ui->actionShow_Bounding_Boxes->setEnabled(sv->renderType()==RT_gl);
    ui->actionShow_Axis->setEnabled(sv->renderType()==RT_gl);
    ui->actionShow_Voxels->setEnabled(sv->renderType()==RT_gl);
    ui->actionShow_Backfaces->setEnabled(sv->renderType()==RT_gl);
    ui->actionTextures_off->setEnabled(sv->renderType()==RT_gl);

    // Menu Ray Tracer
    ui->actionConstant_Redering->setChecked(sv->raytracer()->continuous());
    ui->actionRender_Distributed_RT_features->setChecked(sv->raytracer()->distributed());

    // Menu Window
    ui->actionFullscreen->setChecked(this->isFullScreen());
    bool singleView = (!_activeGLWidget ||
                       !_activeGLWidget->parentWidget() ||
                       !_activeGLWidget->parentWidget()->parentWidget() ||
                        _activeGLWidget->parentWidget()->parentWidget()->parentWidget()==this);
    ui->actionSingle_view->setEnabled(!singleView);
    ui->actionDelete_active_view->setEnabled(!singleView);

    update();
    updateAllGLWidgets();

//   cout << "--------------------" << endl;
//   this->centralWidget()->dumpObjectTree();
//   cout << "--------------------" << endl;
}
//-----------------------------------------------------------------------------
void qtMainWindow::buildNodeTree()
{
    ui->nodeTree->clear();

    if (SLScene::current->root3D())
        addNodeTreeItem(SLScene::current->root3D(), ui->nodeTree, 0);
}
//-----------------------------------------------------------------------------
void qtMainWindow::addNodeTreeItem(SLNode* node,
                                   QTreeWidget* tree,
                                   qtNodeTreeItem* parent)
{
    qtNodeTreeItem* item;
    if (parent) item = new qtNodeTreeItem(node, parent);
    else item = new qtNodeTreeItem(node, tree);

    for (auto m : node->meshes())
        qtNodeTreeItem* mesh = new qtNodeTreeItem(m, item);

    for (auto child : node->children())
        addNodeTreeItem(child, tree, item);
}
//-----------------------------------------------------------------------------
void qtMainWindow::buildPropertyTree()
{
    ui->propertyTree->clear();

    if (!_selectedNodeItem)
    {  ui->propertyTree->update();
        return;
    }
    ui->propertyTree->update();
    ui->propertyTree->setUpdatesEnabled(false); // do this for performance

    qtPropertyTreeWidget::isBeingBuilt = true;
    qtPropertyTreeItem *level1, *level2, *level3;
    SLNode* node = 0;
    bool allowEdit = false;

    // Show Node Properties
    if (_selectedNodeItem->node())
    {  
        node = _selectedNodeItem->node();

        level1 = new qtPropertyTreeItem("Node Name:", "", allowEdit);
        level1->setGetString(bind((const string&(SLNode::*)(void)const)&SLNode::name, node),
                            bind((void(SLNode::*)(const string&))&SLNode::name, node, _1));
        ui->propertyTree->addTopLevelItem(level1);
      
        level1 = new qtPropertyTreeItem("No. of child nodes:",
                                        QString::number(node->children().size()));
        ui->propertyTree->addTopLevelItem(level1);
      
        level1 = new qtPropertyTreeItem("No. of child meshes:",
                                        QString::number(node->meshes().size()));
        ui->propertyTree->addTopLevelItem(level1);

        // Get the object transform matrix
        SLMat4f om(node->om());
        SLVec3f translation, rotAngles, scaleFactors;
        om.decompose(translation, rotAngles, scaleFactors);
        rotAngles *= SL_RAD2DEG;
      
        level1 = new qtPropertyTreeItem("Local Transform:");
        ui->propertyTree->addTopLevelItem(level1);

        level2 = new qtPropertyTreeItem("Translation:", QString::fromStdString(translation.toString()));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Rotation:", QString::fromStdString(rotAngles.toString()));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Scaling:", QString::fromStdString(scaleFactors.toString()));
        level1->addChild(level2);

      
        // Add Drawing flags sub items
        level1 = new qtPropertyTreeItem("Drawflags:");
        ui->propertyTree->addTopLevelItem(level1);

        level2 = new qtPropertyTreeItem("Hide:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_HIDDEN),
                           bind((void(SLNode::*)(uint, bool))&SLNode::setDrawBitsRec, node, SL_DB_HIDDEN, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Normals:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_NORMALS),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_NORMALS, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Wire Mesh:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_WIREMESH),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_WIREMESH, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Bounding Box:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_BBOX),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_BBOX, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Axis:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_AXIS),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_AXIS, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Skeleton:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_SKELETON),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_SKELETON, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Voxels:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_VOXELS),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_VOXELS, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Show Back Faces:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_CULLOFF),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_CULLOFF, _1));
        level1->addChild(level2);

        level2 = new qtPropertyTreeItem("Textures off:", "", true);
        level2->setGetBool(bind((bool(SLNode::*)(uint))&SLNode::drawBit, node, SL_DB_TEXOFF),
                           bind(&SLNode::setDrawBitsRec, node, SL_DB_TEXOFF, _1));
        level1->addChild(level2);

      
        // Show special camera properties
        if (typeid(*node)==typeid(SLCamera))
        {
            SLCamera* cam = (SLCamera*)node;
            if (cam == _activeGLWidget->sv()->camera())
                 level1 = new qtPropertyTreeItem("Camera (active):");
            else level1 = new qtPropertyTreeItem("Camera:");
            ui->propertyTree->addTopLevelItem(level1);

            level2 = new qtPropertyTreeItem("Field of view:", "", allowEdit);
            level2->setGetFloat(bind((float(SLCamera::*)(void) const)&SLCamera::fov, cam),
                                bind((void(SLCamera::*)(float))&SLCamera::fov, cam, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Near clip plane:", "", allowEdit);
            level2->setGetFloat(bind((float(SLCamera::*)(void) const)&SLCamera::clipNear, cam),
                                bind((void(SLCamera::*)(float))&SLCamera::clipNear, cam, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Far clip plane:", "", allowEdit);
            level2->setGetFloat(bind((float(SLCamera::*)(void) const)&SLCamera::clipFar, cam),
                                bind((void(SLCamera::*)(float))&SLCamera::clipFar, cam, _1));
            level1->addChild(level2);
        }

        // Show special light properties
        if (typeid(*node)==typeid(SLLightSphere) || typeid(*node)==typeid(SLLightRect))
        {
            SLLight* light;
            if (typeid(*node)==typeid(SLLightSphere)) light = (SLLight*)(SLLightSphere*)node;
            if (typeid(*node)==typeid(SLLightRect)) light = (SLLight*)(SLLightRect*)node;

            level1 = new qtPropertyTreeItem("Light:");
            ui->propertyTree->addTopLevelItem(level1);

            level2 = new qtPropertyTreeItem("Turned on:", "", true);
            level2->setGetBool(bind((bool(SLLight::*)(void))&SLLight::on, light),
                               bind((void(SLLight::*)(bool))&SLLight::on, light, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Ambient Intensity:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLLight::*)(void))&SLLight::ambient, light),
                                bind((void(SLLight::*)(SLCol4f))&SLLight::ambient, light, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Diffuse Intensity:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLLight::*)(void))&SLLight::diffuse, light),
                                bind((void(SLLight::*)(SLCol4f))&SLLight::diffuse, light, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Specular Intensity:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLLight::*)(void))&SLLight::specular, light),
                                bind((void(SLLight::*)(SLCol4f))&SLLight::specular, light, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Cut off angle:", "", allowEdit);
            level2->setGetFloat(bind((float(SLLight::*)(void))&SLLight::spotCutoff, light),
                                bind((void(SLLight::*)(float))&SLLight::spotCutoff, light, _1));
            level1->addChild(level2);

        }
    }

    // Show Mesh Properties
    if (_selectedNodeItem->mesh())
    {
        SLMesh* mesh = _selectedNodeItem->mesh();
        SLMaterial* mat = mesh->mat;

        level1 = new qtPropertyTreeItem("Mesh Name:", "", allowEdit);
        level1->setGetString(bind((const string&(SLMesh::*)(void) const)&SLMesh::name, mesh),
                             bind((void(SLMesh::*)(const string&))&SLMesh::name, mesh, _1));
        ui->propertyTree->addTopLevelItem(level1);

        if (mesh->primitive()==PT_triangles)
            level1 = new qtPropertyTreeItem("Vertices/Triangles",
                                            QString::number(mesh->numV)+" / "+
                                            QString::number(mesh->numI/3));
        if (mesh->primitive()==PT_lines)
            level1 = new qtPropertyTreeItem("Vertices/Lines",
                                            QString::number(mesh->numV)+" / "+
                                            QString::number(mesh->numI/2));
        ui->propertyTree->addTopLevelItem(level1);

        if (mat)
        {
            level1 = new qtPropertyTreeItem("Material:", "", allowEdit);
            level1->setGetString(bind((const string&(SLMaterial::*)(void)const)&SLMaterial::name, mat),
                                 bind((void(SLMaterial::*)(const string&))&SLMaterial::name, mat, _1));
            ui->propertyTree->addTopLevelItem(level1);

            level2 = new qtPropertyTreeItem("Ambient Color:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLMaterial::*)(void))&SLMaterial::ambient, mat),
                                bind((void(SLMaterial::*)(SLCol4f))&SLMaterial::ambient, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Diffuse Color:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLMaterial::*)(void))&SLMaterial::diffuse, mat),
                                bind((void(SLMaterial::*)(SLCol4f))&SLMaterial::diffuse, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Specular Color:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLMaterial::*)(void))&SLMaterial::specular, mat),
                                bind((void(SLMaterial::*)(SLCol4f))&SLMaterial::specular, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Emissive Color:", "", allowEdit);
            level2->setGetVec4f(bind((SLCol4f(SLMaterial::*)(void))&SLMaterial::emission, mat),
                                bind((void(SLMaterial::*)(SLCol4f))&SLMaterial::emission, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Shininess:", "", allowEdit);
            level2->setGetFloat(bind((float(SLMaterial::*)(void))&SLMaterial::shininess, mat),
                                bind((void(SLMaterial::*)(float))&SLMaterial::shininess, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Reflectivity:", "", allowEdit);
            level2->setGetFloat(bind((float(SLMaterial::*)(void))&SLMaterial::kr, mat),
                                bind((void(SLMaterial::*)(float))&SLMaterial::kr, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Transparency:", "", allowEdit);
            level2->setGetFloat(bind((float(SLMaterial::*)(void))&SLMaterial::kt, mat),
                                bind((void(SLMaterial::*)(float))&SLMaterial::kt, mat, _1));
            level1->addChild(level2);

            level2 = new qtPropertyTreeItem("Refractive Index:", "", allowEdit);
            level2->setGetFloat(bind((float(SLMaterial::*)(void))&SLMaterial::kn, mat),
                                bind((void(SLMaterial::*)(float))&SLMaterial::kn, mat, _1));
            level1->addChild(level2);


            SLGLProgram *prog = mat->program();
            if (prog)
            {  level2 = new qtPropertyTreeItem("Shader Program:");
                level1->addChild(level2);

                SLVGLShader& shaders = prog->shaders();
                for (auto shader : shaders)
                {   if(shader->shaderType() ==ST_vertex)
                    {   level3 = new qtPropertyTreeItem("Vertex Shader:");
                        level3->setGetString(bind((const string&(SLGLShader::*)(void)const)&SLGLShader::name, shader),
                                             bind((void(SLGLShader::*)(const string&))&SLGLShader::name, shader, _1));
                        level2->addChild(level3);
                    } else
                    if(shader->shaderType()==ST_fragment)
                    {   level3 = new qtPropertyTreeItem("Fragment Shader:");
                        level3->setGetString(bind((const string&(SLGLShader::*)(void)const)&SLGLShader::name, shader),
                                             bind((void(SLGLShader::*)(const string&))&SLGLShader::name, shader, _1));
                        level2->addChild(level3);
                    }
                }
            }

            if (mat->textures().size() > 0)
            {   level2 = new qtPropertyTreeItem("Textures:");
                level1->addChild(level2);

                for (auto texture : mat->textures())
                {   level3 = new qtPropertyTreeItem("Texture:");
                    level3->setGetString(bind((const string&(SLGLTexture::*)(void)const)&SLGLTexture::name, texture),
                                         bind((void(SLGLTexture::*)(const string&))&SLGLTexture::name, texture, _1));
                    level2->addChild(level3);
                }
            }
        }
    }
   

    qtPropertyTreeWidget::isBeingBuilt = false;
    ui->propertyTree->setUpdatesEnabled(true); // do this for performance
    ui->propertyTree->update();
}
//-----------------------------------------------------------------------------
void qtMainWindow::updateAnimationList()
{
    // clear both lists
    SLbool hasAnimations = false;
    ui->animAnimatedObjectSelect->clear();
    ui->animAnimationSelect->clear();

    ui->animAnimatedObjectSelect->addItem("Select Target", -1);
    ui->animAnimationSelect->addItem("Select animation", -1);

    _selectedAnim = NULL;
    SLVSkeleton& skeletons = SLScene::current->animManager().skeletons();
    
    if (SLScene::current->animManager().animations().size() > 0)
    {   ui->animAnimatedObjectSelect->addItem("Node Animations", 0);
        hasAnimations = true;
    }

    for (SLint i = 0; i < skeletons.size(); ++i)
    {   SLint index = ui->animAnimatedObjectSelect->count();
        ui->animAnimatedObjectSelect->addItem("Skeleton " + QString::number(i), i+1);
        hasAnimations = true;
    }

    if (hasAnimations)
    {   ui->animAnimatedObjectSelect->setCurrentIndex(1); // select first item        
        ui->dockAnimation->show();
    } else
    {   // hide the animation ui element completely since we don't need it
        ui->dockAnimation->hide();
    }

}

//-----------------------------------------------------------------------------
void qtMainWindow::updateAnimationTimeline()
{
    if (!_selectedAnim)
        return;
    
    ui->animTimelineSlider->setCurrentTime(_selectedAnim->localTime());
    ui->animCurrentTimeLabel->setText(ui->animTimelineSlider->getCurrentTimeString());
}

//-----------------------------------------------------------------------------
void qtMainWindow::selectAnimationFromNode(SLNode* node)
{
    for(auto& kv : SLScene::current->animManager().animations())
    {
        if (kv.second->affectsNode(node))
        {
            // select node animations
            ui->animAnimatedObjectSelect->setCurrentIndex(1);

            // find and select correct animation
            SLAnimPlayback* play = SLScene::current->animManager().getNodeAnimPlayack(kv.second->name());
            QVariant variant;
            variant.setValue<SLAnimPlayback*>(play);
            SLint index = ui->animAnimationSelect->findData(variant);
            ui->animAnimationSelect->setCurrentIndex(index);
        }
    }

    for (auto mesh : node->meshes())
    {
        if (!mesh->skeleton())
            continue;

        SLint selectIndex = 1;
        for (auto skeleton : SLScene::current->animManager().skeletons())
        {
            // find and select the skeleton
            if (mesh->skeleton() == skeleton)
                ui->animAnimatedObjectSelect->setCurrentIndex(ui->animAnimatedObjectSelect->findData(selectIndex));
            ++selectIndex;
        }
    }
}

//-----------------------------------------------------------------------------
void qtMainWindow::beforeSceneLoad()
{
    //on_actionSingle_view_triggered();
    _selectedNodeItem = 0;
    ui->nodeTree->clear();
    ui->propertyTree->clear();
    updateAnimationList();
}
//-----------------------------------------------------------------------------
void qtMainWindow::afterSceneLoad()
{
    setMenuState();
    buildNodeTree();
    updateAnimationList();
}
//-----------------------------------------------------------------------------
void qtMainWindow::selectNodeOrMeshItem(SLNode* selectedNode, SLMesh* selectedMesh)
{
    if (selectedNode==0)
    {   ui->propertyTree->clear();
        ui->nodeTree->collapseAll();
        return;
    }


    // select animation related to this node if it exists
    selectAnimationFromNode(selectedNode);

    QTreeWidgetItemIterator it(ui->nodeTree);
    while (*it) 
    {
        qtNodeTreeItem* item = (qtNodeTreeItem*)*it;
        if (item->mesh() == selectedMesh &&
            item->node() == selectedNode)
        {   if (_selectedNodeItem)
                if (_selectedNodeItem!=item)
                    _selectedNodeItem->setSelected(false);
            item->setSelected(true);
            _selectedNodeItem = item;

            // expand tree
            ui->nodeTree->collapseAll();
            QTreeWidgetItem* parent = item->parent();
            while(parent)
            {  parent->setExpanded(true);
            parent = parent->parent();
            }

            ui->nodeTree->scrollToItem(item);
            ui->nodeTree->update();
            buildPropertyTree();
            updateAllGLWidgets();
            return;
        }
        ++it;
    }
}
//-----------------------------------------------------------------------------
void qtMainWindow::updateAllGLWidgets()
{
    for (auto widget : _allGLWidgets) widget->update();
    updateAnimationTimeline();
}
//-----------------------------------------------------------------------------
void qtMainWindow::applyCommandOnSV(const SLCommand cmd)
{
    if (QApplication::keyboardModifiers() == Qt::ShiftModifier)
        SLScene::current->onCommandAllSV(cmd);
    else _activeGLWidget->sv()->onCommand(cmd);
    setMenuState();
}

//-----------------------------------------------------------------------------
//! Returns the other QGLWidget in the same splitter
qtGLWidget* qtMainWindow::getOtherGLWidgetInSplitter()
{
    if (!_activeGLWidget || 
        !_activeGLWidget->parentWidget() ||
        !_activeGLWidget->parentWidget()->parentWidget() ||
        _activeGLWidget->parentWidget()->parentWidget()->parentWidget()==this)
    {   return 0;
    }

    QSplitter* parentSplitter = (QSplitter*)_activeGLWidget->parentWidget()->parentWidget()->parentWidget();
    QSplitter* activeSplitter = (QSplitter*)_activeGLWidget->parentWidget()->parentWidget();
    QSplitter* otherSplitter;

    // Get other splitter
    for (int i=0; i < parentSplitter->count(); ++i)
    {   if (parentSplitter->widget(i) != activeSplitter)
        {  if (typeid(*parentSplitter->widget(i)) == typeid(QSplitter))
            {  otherSplitter = (QSplitter*)parentSplitter->widget(i);
            }
        }
    }

    // Set the next active GLWidget within the otherSplitter
    qtGLWidget* otherGLWidget = (qtGLWidget*)otherSplitter->findChild<QGLWidget*>();
    return otherGLWidget;
}

// Overwritten Event Handlers
//-----------------------------------------------------------------------------
void qtMainWindow::resizeEvent(QResizeEvent* event)
{   
    setMenuState();
}
void qtMainWindow::changeEvent(QEvent* event)
{
    if(event->type() == QEvent::WindowStateChange)
    {
        bool isFull = isFullScreen();
        ui->menuBar->setVisible(!isFull);
        ui->statusBar->setVisible(!isFull);
        ui->toolBar->setVisible(!isFull);
        setMenuState();
    }
}
//-----------------------------------------------------------------------------
void qtMainWindow::closeEvent(QCloseEvent *event)
{  
    slTerminate();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// ACTIONS
//
// Menu File
void qtMainWindow::on_action_Quit_triggered()
{
    SLScene::current->unInit();
    delete SLScene::current;
    QApplication::exit(0);
}

// Menu Load Scene
void qtMainWindow::on_actionSmall_Test_Scene_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneMinimal);
    afterSceneLoad();
}
void qtMainWindow::on_actionLarge_Model_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneLargeModel);
    afterSceneLoad();
}
void qtMainWindow::on_actionFigure_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneFigure);
    afterSceneLoad();
}
void qtMainWindow::on_actionMesh_Loader_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneMeshLoad);
    afterSceneLoad();
}
void qtMainWindow::on_actionTexture_Blending_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneTextureBlend);
    afterSceneLoad();
}
void qtMainWindow::on_actionTexture_Filtering_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneTextureFilter);
    afterSceneLoad();
}
void qtMainWindow::on_actionFrustum_Culling_1_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneFrustumCull1);
    afterSceneLoad();
}

void qtMainWindow::on_actionPer_Vertex_Lighting_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_scenePerVertexBlinn);
    afterSceneLoad();
}
void qtMainWindow::on_actionPer_Pixel_Lighting_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_scenePerPixelBlinn);
    afterSceneLoad();
}
void qtMainWindow::on_actionPer_Vertex_Wave_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_scenePerVertexWave);
    afterSceneLoad();
}
void qtMainWindow::on_actionWater_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneWater);
    afterSceneLoad();
}
void qtMainWindow::on_actionBump_Mapping_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneBumpNormal);
    afterSceneLoad();
}
void qtMainWindow::on_actionParallax_Mapping_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneBumpParallax);
    afterSceneLoad();
}
void qtMainWindow::on_actionGlass_Shader_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneRevolver);
    afterSceneLoad();
}
void qtMainWindow::on_actionEarth_Shader_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneEarth);
    afterSceneLoad();
}

void qtMainWindow::on_actionNode_Animation_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneNodeAnimation);
    afterSceneLoad();
}
void qtMainWindow::on_actionSkeletal_Animation_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneSkeletalAnimation);
    afterSceneLoad();
}
void qtMainWindow::on_actionAstroboy_Army_CPU_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneAstroboyArmyCPU);
    afterSceneLoad();
}
void qtMainWindow::on_actionAstroboy_Army_GPU_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneAstroboyArmyGPU);
    afterSceneLoad();
}
void qtMainWindow::on_actionMass_Animation_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneMassAnimation);
    afterSceneLoad();
}

void qtMainWindow::on_actionRT_Spheres_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneRTSpheres);
    afterSceneLoad();
}
void qtMainWindow::on_actionRT_Muttenzer_Box_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneRTMuttenzerBox);
    afterSceneLoad();
}
void qtMainWindow::on_actionRT_Depth_of_Field_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneRTDoF);
    afterSceneLoad();
}
void qtMainWindow::on_actionRT_Lens_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneRTLens);
    afterSceneLoad();
}
void qtMainWindow::on_actionRT_Soft_Shadows_triggered()
{
    beforeSceneLoad();
    _activeGLWidget->sv()->onCommand(C_sceneRTSoftShadows);
    afterSceneLoad();
}

// Menu Renderer
void qtMainWindow::on_actionOpenGL_triggered()
{
    _activeGLWidget->sv()->onCommand(C_renderOpenGL);
    setMenuState();
    updateAllGLWidgets();
}
void qtMainWindow::on_actionRay_Tracer_triggered()
{
    SLSceneView* sv =  _activeGLWidget->sv();
    if (sv->camera()->projection() > P_monoOrthographic)
        sv->onCommand(C_projPersp);
    sv->onCommand(C_rt5);
    setMenuState();
}
void qtMainWindow::on_actionPath_Tracer_triggered()
{
    SLSceneView* sv =  _activeGLWidget->sv();
    if (sv->camera()->projection() > P_monoOrthographic)
        sv->onCommand(C_projPersp);
    sv->onCommand(C_pt10);
    setMenuState();
}

// Menu Info
void qtMainWindow::on_actionShow_DockScenegraph_triggered()
{
    if (ui->dockScenegraph->isVisible())
        ui->dockScenegraph->hide();
    else 
        ui->dockScenegraph->show();
}
void qtMainWindow::on_actionShow_DockProperties_triggered()
{
    if (ui->dockProperties->isVisible())
        ui->dockProperties->hide();
    else 
        ui->dockProperties->show();
}
void qtMainWindow::on_actionShow_Animation_Controler_triggered()
{
    if (ui->dockAnimation->isVisible())
        ui->dockAnimation->hide();
    else
        ui->dockAnimation->show();
}
void qtMainWindow::on_actionShow_Toolbar_triggered()
{
    ui->toolBar->setVisible(!ui->toolBar->isVisible());
}
void qtMainWindow::on_actionShow_Statusbar_triggered()
{
    ui->statusBar->setVisible(!ui->statusBar->isVisible());
}
void qtMainWindow::on_actionShow_Statistics_triggered()
{
    applyCommandOnSV(C_statsToggle);
}
void qtMainWindow::on_actionShow_Scene_Info_triggered()
{
    applyCommandOnSV(C_sceneInfoToggle);
}
void qtMainWindow::on_actionShow_Menu_triggered()
{
    SLSceneView* sv = _activeGLWidget->sv();
    sv->showMenu(!sv->showMenu());
    setMenuState();
}
void qtMainWindow::on_actionAbout_SLProject_triggered()
{
    QMessageBox::information(this, "About SLProject",
                             QString::fromStdString(SLScene::current->infoAbout_en()));
}
void qtMainWindow::on_actionCredits_triggered()
{
    QMessageBox::information(this, "About External Libraries",
                             QString::fromStdString(SLScene::current->infoCredits_en()).replace("\\n","\n"));
}
void qtMainWindow::on_actionAbout_Qt_triggered()
{
    QMessageBox::aboutQt(this, "About Qt");
}

// Menu Camera
void qtMainWindow::on_actionReset_triggered()
{
    applyCommandOnSV(C_camReset);
}
void qtMainWindow::on_actionUse_SceneView_Camera_triggered()
{
    applyCommandOnSV(C_useSceneViewCamera);
}
void qtMainWindow::on_actionPerspective_triggered()
{
    applyCommandOnSV(C_projPersp);
}
void qtMainWindow::on_actionOrthographic_triggered()
{
    applyCommandOnSV(C_projOrtho);
}
void qtMainWindow::on_actionSide_by_side_triggered()
{
    applyCommandOnSV(C_projSideBySide);
}
void qtMainWindow::on_actionSide_by_side_proportional_triggered()
{
    applyCommandOnSV(C_projSideBySideP);
}
void qtMainWindow::on_actionSide_by_side_distorted_triggered()
{
    applyCommandOnSV(C_projSideBySideD);
}
void qtMainWindow::on_actionLine_by_line_triggered()
{
    applyCommandOnSV(C_projLineByLine);
}
void qtMainWindow::on_actionColumn_by_column_triggered()
{
    applyCommandOnSV(C_projColumnByColumn);
}
void qtMainWindow::on_actionPixel_by_pixel_triggered()
{
    applyCommandOnSV(C_projPixelByPixel);
}
void qtMainWindow::on_actionColor_Red_Cyan_triggered()
{
    applyCommandOnSV(C_projColorRC);
}
void qtMainWindow::on_actionColor_Red_Green_triggered()
{
    applyCommandOnSV(C_projColorRG);
}
void qtMainWindow::on_actionColor_Red_Blue_triggered()
{
    applyCommandOnSV(C_projColorRB);
}
void qtMainWindow::on_actionColor_Cyan_Yellow_triggered()
{
    applyCommandOnSV(C_projColorYB);
}
void qtMainWindow::on_action_eyeSepInc10_triggered()
{
    applyCommandOnSV(C_camEyeSepInc);
}
void qtMainWindow::on_action_eyeSepDec10_triggered()
{
    applyCommandOnSV(C_camEyeSepDec);
}
void qtMainWindow::on_action_focalDistInc_triggered()
{
    applyCommandOnSV(C_camFocalDistInc);
}
void qtMainWindow::on_action_focalDistDec_triggered()
{
    applyCommandOnSV(C_camFocalDistDec);
}
void qtMainWindow::on_action_fovInc10_triggered()
{
    applyCommandOnSV(C_camFOVInc);
}
void qtMainWindow::on_action_fovDec10_triggered()
{
    applyCommandOnSV(C_camFOVDec);
}
void qtMainWindow::on_actionTurntable_Y_up_triggered()
{
    applyCommandOnSV(C_camAnimTurnYUp);
}
void qtMainWindow::on_actionTurntable_Z_up_triggered()
{
    applyCommandOnSV(C_camAnimTurnZUp);
}
void qtMainWindow::on_actionWalking_Y_up_triggered()
{
    applyCommandOnSV(C_camAnimWalkYUp);
}
void qtMainWindow::on_actionWalking_Z_up_triggered()
{
    applyCommandOnSV(C_camAnimWalkZUp);
}
void qtMainWindow::on_action_speedInc_triggered()
{
    applyCommandOnSV(C_camSpeedLimitInc);
}
void qtMainWindow::on_action_speedDec_triggered()
{
    applyCommandOnSV(C_camSpeedLimitDec);
}

// Menu Render Flags
void qtMainWindow::on_actionDepthTest_triggered()
{
    applyCommandOnSV(C_depthTestToggle);
}
void qtMainWindow::on_actionAntialiasing_triggered()
{
    applyCommandOnSV(C_multiSampleToggle);
}
void qtMainWindow::on_actionView_Frustum_Culling_triggered()
{
    applyCommandOnSV(C_frustCullToggle);
}
void qtMainWindow::on_actionSlowdown_on_Idle_triggered()
{
    applyCommandOnSV(C_waitEventsToggle);
}

void qtMainWindow::on_actionShow_Normals_triggered()
{
    applyCommandOnSV(C_normalsToggle);
}
void qtMainWindow::on_actionShow_Wired_Mesh_triggered()
{
    applyCommandOnSV(C_wireMeshToggle);
}
void qtMainWindow::on_actionShow_Bounding_Boxes_triggered()
{
    applyCommandOnSV(C_bBoxToggle);
}
void qtMainWindow::on_actionShow_Axis_triggered()
{
    applyCommandOnSV(C_axisToggle);
}
void qtMainWindow::on_actionShow_Skeleton_triggered()
{
    applyCommandOnSV(C_skeletonToggle);
}
void qtMainWindow::on_actionShow_Backfaces_triggered()
{
    applyCommandOnSV(C_faceCullToggle);
}
void qtMainWindow::on_actionShow_Voxels_triggered()
{
    applyCommandOnSV(C_voxelsToggle);
}
void qtMainWindow::on_actionTextures_off_triggered()
{
    applyCommandOnSV(C_textureToggle);
}

void qtMainWindow::on_actionAnimation_off_triggered()
{
    applyCommandOnSV(C_animationToggle);
    ui->dockAnimation->setEnabled(!ui->actionAnimation_off->isChecked());
}

// Menu Ray Tracing
void qtMainWindow::on_actionRender_to_depth_1_triggered()
{
    applyCommandOnSV(C_rt1);
}
void qtMainWindow::on_actionRender_to_depth_2_triggered()
{
    applyCommandOnSV(C_rt2);
}
void qtMainWindow::on_actionRender_to_depth_5_triggered()
{
    applyCommandOnSV(C_rt5);
}
void qtMainWindow::on_actionRender_to_max_depth_triggered()
{
    applyCommandOnSV(C_rt0);
}
void qtMainWindow::on_actionConstant_Redering_triggered()
{
    applyCommandOnSV(C_rtContinuously);
    setMenuState();
}
void qtMainWindow::on_actionRender_Distributed_RT_features_triggered()
{
    applyCommandOnSV(C_rtDistributed);
    setMenuState();
}

// Menu Path Tracing
void qtMainWindow::on_action1_Sample_triggered()
{
    applyCommandOnSV(C_pt1);
}
void qtMainWindow::on_action10_Samples_triggered()
{
    applyCommandOnSV(C_pt10);
}
void qtMainWindow::on_action100_Sample_triggered()
{
    applyCommandOnSV(C_pt100);
}
void qtMainWindow::on_action1000_Samples_triggered()
{
    applyCommandOnSV(C_pt1000);
}
void qtMainWindow::on_action10000_Samples_triggered()
{
    applyCommandOnSV(C_pt10000);
}

// Menu Window
void qtMainWindow::on_actionFullscreen_triggered()
{
    // See also changeEvent where menu-, tool- & statusbar get hidden
    if (isFullScreen())
         setWindowState(Qt::WindowNoState);
    else setWindowState(Qt::WindowFullScreen);
}
void qtMainWindow::on_actionSplit_active_view_horizontally_triggered()
{
    qtGLWidget* newWidget = _activeGLWidget->splitActive(Qt::Horizontal);
    setMenuState();
}
void qtMainWindow::on_actionSplit_active_view_vertically_triggered()
{
    qtGLWidget* newWidget = _activeGLWidget->splitActive(Qt::Vertical);
    setMenuState();
}
void qtMainWindow::on_actionSplit_into_4_views_triggered()
{
    // top right view views scene from top
    qtGLWidget* topRight = _activeGLWidget->splitActive(Qt::Horizontal);
    qtGLWidget* bottomLeft = _activeGLWidget->splitActive(Qt::Vertical);
    qtGLWidget* bottomRight = topRight->splitActive(Qt::Vertical);

    // manually call init to set the correct scene view camera positions
    topRight->sv()->initSceneViewCamera(-SLVec3f::AXISY, P_monoOrthographic);
    bottomLeft->sv()->initSceneViewCamera(-SLVec3f::AXISZ, P_monoOrthographic);
    bottomRight->sv()->initSceneViewCamera(-SLVec3f::AXISX, P_monoOrthographic);

    setMenuState();
}
void qtMainWindow::on_actionSingle_view_triggered()
{
    // keep the active widget
    qtGLWidget* oldActiveGLWidget = _activeGLWidget;
    _activeGLWidget = getOtherGLWidgetInSplitter();

    // delete the other until only the former active remains
    while (_activeGLWidget)
    {
        on_actionDelete_active_view_triggered();
        _activeGLWidget = getOtherGLWidgetInSplitter();
        if (_activeGLWidget==oldActiveGLWidget)
            _activeGLWidget = getOtherGLWidgetInSplitter();
    }
    _activeGLWidget = oldActiveGLWidget;
}
void qtMainWindow::on_actionDelete_active_view_triggered()
{
    // Example1: The composition of widgets before the delete:
    //
    // +--------------------------------------------------------------------+
    // | parentSplitter                                                     |
    // | +---------------------------+   +--------------------------------+ |
    // | | activeSplitter            |   | ohterSplitter                  | |
    // | | ######################### |   | ############################## | |
    // | | # borderWidget1         # |   | # borderWidget2              # | |
    // | | # +-------------------+ # |   | # +------------------------+ # | |
    // | | # | activeGLWidget1   | # |   | # | QGLWidget2             | # | |  
    // | | # |                   | # |   | # |                        | # | |  
    // | | # |                   | # | O | # |                        | # | |  
    // | | # |                   | # |   | # |                        | # | |  
    // | | # |                   | # |   | # |                        | # | |  
    // | | # +-------------------+ # |   | # +------------------------+ # | |  
    // | | ######################### |   | ############################## | |
    // | +---------------------------+   +--------------------------------+ |
    // +--------------------------------------------------------------------+
    //
    // The composition of after the delete:
    //
    // +--------------------------------------------------------------------+
    // | parentSplitter                                                     |
    // | ################################################################## |
    // | # borderWidget2 for frame                                        # |
    // | # +------------------------------------------------------------+ # |
    // | # | activeGLWidget2                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # |                                                            | # |
    // | # +------------------------------------------------------------+ # |
    // | ################################################################## |
    // +--------------------------------------------------------------------+
    //
    //
    // Example2: The composition of widgets before the delete:
    //
    // +--------------------------------------------------------------------+
    // | parentSplitter                                                     |
    // | +---------------------------+   +--------------------------------+ |
    // | | otherSplitter             |   | activeSplitter                 | |
    // | | +-----------------------+ |   | ############################## | |
    // | | | splitter3             | |   | # borderWidget               # | |
    // | | | ##################### | |   | # +------------------------+ # | |
    // | | | # borderWidget3     # | |   | # | activeGLWidget         | # | |   
    // | | | # +---------------+ # | |   | # |                        | # | |  
    // | | | # | GLWidget1     | # | |   | # |                        | # | |  
    // | | | # |               | # | |   | # |                        | # | |  
    // | | | # |               | # | |   | # |                        | # | | 
    // | | | # +---------------+ # | |   | # |                        | # | |  
    // | | | ##################### | |   | # |                        | # | |  
    // | | +-----------------------+ |   | # |                        | # | |  
    // | |             O             | O | # |                        | # | |  
    // | | +-----------------------+ |   | # |                        | # | |
    // | | | splitter4             | |   | # |                        | # | |  
    // | | | ##################### | |   | # |                        | # | |  
    // | | | # borderWidget4     # | |   | # |                        | # | |  
    // | | | # +---------------+ # | |   | # |                        | # | |  
    // | | | # | GLWidget4     | # | |   | # |                        | # | |  
    // | | | # |               | # | |   | # |                        | # | |  
    // | | | # |               | # | |   | # |                        | # | |
    // | | | # +---------------+ # | |   | # |                        | # | |
    // | | | ##################### | |   | # +------------------------+ # | |
    // | | +-----------------------+ |   | ############################## | |
    // | +---------------------------+   +--------------------------------+ |
    // +--------------------------------------------------------------------+
    //
    // The composition of after the delete:
    //
    // +--------------------------------------------------------------------+
    // | parentSplitter                                                     |
    // | +----------------------------------------------------------------+ |
    // | | splitter3                                                      | |
    // | | ############################################################## | |
    // | | # borderWidget3                                              # | |   
    // | | # +--------------------------------------------------------+ # | |  
    // | | # | GLWidget1                                              | # | |  
    // | | # |                                                        | # | |  
    // | | # |                                                        | # | | 
    // | | # +--------------------------------------------------------+ # | |  
    // | | ############################################################## | |  
    // | +----------------------------------------------------------------+ |  
    // |                                   O                                |  
    // | +----------------------------------------------------------------+ |
    // | | splitter4                                                      | |  
    // | | ############################################################## | |  
    // | | # borderWidget4                                              # | |  
    // | | # +--------------------------------------------------------+ # | |  
    // | | # | GLWidget4                                              | # | |  
    // | | # |                                                        | # | |  
    // | | # |                                                        | # | |
    // | | # +--------------------------------------------------------+ # | |
    // | | ############################################################## | |
    // | +----------------------------------------------------------------+ |
    // +--------------------------------------------------------------------+

    if (!_activeGLWidget || 
        !_activeGLWidget->parentWidget() ||
        !_activeGLWidget->parentWidget()->parentWidget() ||
        _activeGLWidget->parentWidget()->parentWidget()->parentWidget()==this)
    {   SL_LOG("deleteActive: Can't delete the first GLWidget\n");
        return;
    }

    QSplitter* parentSplitter = (QSplitter*)_activeGLWidget->parentWidget()->parentWidget()->parentWidget();
    QSplitter* activeSplitter = (QSplitter*)_activeGLWidget->parentWidget()->parentWidget();
    QSplitter* otherSplitter;   

    // Delete splitter widget of activeGLWidget
    activeSplitter->setParent(0);
    delete _activeGLWidget->sv();
    delete activeSplitter;
    _allGLWidgets.erase(std::remove(_allGLWidgets.begin(), _allGLWidgets.end(), _activeGLWidget), _allGLWidgets.end());

    // Get other widget
    for (int i=0; i < parentSplitter->count(); ++i)
    {   if (parentSplitter->widget(i) != activeSplitter)
        {  if (typeid(*parentSplitter->widget(i)) == typeid(QSplitter))
            {  otherSplitter = (QSplitter*)parentSplitter->widget(i);
            }
        }
    }

    // Set the next active GLWidget within the otherSplitter
    _activeGLWidget = (qtGLWidget*)otherSplitter->findChild<QGLWidget*>();

    // Set active SceneView and make active border red
    if (_activeGLWidget)
    {   if (_activeGLWidget->parent()->isWidgetType())
        {  QWidget* borderWidget = (QWidget*)_activeGLWidget->parent();
            borderWidget->setStyleSheet("border:2px solid red;");
        }
    } else return;
            
    // Get all children of the otherSplitter into a vector
    int numOtherChildren = otherSplitter->count();
    std::vector<QWidget*> otherWidgets;
    Qt::Orientation otherOrientation =  otherSplitter->orientation();
    for (int i=0; i < otherSplitter->count(); ++i)
        otherWidgets.push_back(otherSplitter->widget(i));

    // remove widgets from other splitter by setting the parent to 0
    for (auto ow : otherWidgets) ow->setParent(0);
   
    // must be 0 now.
    numOtherChildren = otherSplitter->count();

    // Delete the obsolete other splitter
    otherSplitter->setParent(0); 
    delete otherSplitter;
   
    // must be 0 now.
    int numParentChildren = parentSplitter->count();

    // Reattach other widgets to the parent
    for (auto ow : otherWidgets) 
        parentSplitter->addWidget(ow);

    // Take over the otherSplitters layout direction
    parentSplitter->setOrientation(otherOrientation);

    setMenuState();
}

// Other Slots
void qtMainWindow::on_nodeTree_itemClicked(QTreeWidgetItem *item, int column)
{
    SLScene* s = SLScene::current;
    qtNodeTreeItem* nodeItem = (qtNodeTreeItem*)item;

    if (item->isSelected())
    {   if (_selectedNodeItem)
            if (_selectedNodeItem!=item)
            _selectedNodeItem->setSelected(false);
        _selectedNodeItem = nodeItem;
        s->selectNodeMesh(_selectedNodeItem->node(), _selectedNodeItem->mesh());
    } else
    {   _selectedNodeItem = 0;
        s->selectNodeMesh(0, 0);
    }
    buildPropertyTree();
    updateAllGLWidgets();
}
void qtMainWindow::on_nodeTree_itemDoubleClicked(QTreeWidgetItem *item, int column)
{
    SLSceneView* sv = _activeGLWidget->sv();
    SLNode* node = ((qtNodeTreeItem*)item)->node();

    // Set active Camera
    if (typeid(*node)==typeid(SLCamera))
        sv->camera((SLCamera*)node);

    updateAllGLWidgets();
    
    // we need to set the menu state because the scene camera might not be active anymore
    setMenuState();
}
void qtMainWindow::on_dockScenegraph_visibilityChanged(bool visible)
{
    setMenuState();
}
void qtMainWindow::on_dockProperties_visibilityChanged(bool visible)
{
    setMenuState();
}
void qtMainWindow::on_dockAnimation_visibilityChanged(bool visible)
{
    setMenuState();
}
void qtMainWindow::on_propertyTree_itemChanged(QTreeWidgetItem *item, int column)
{
    ((qtPropertyTreeItem*)item)->onItemChanged(column);
    ui->propertyTree->update();
    updateAllGLWidgets();
}

// Animation Controller
void qtMainWindow::on_animAnimatedObjectSelect_currentIndexChanged(int index)
{
    int data = ui->animAnimatedObjectSelect->itemData(index).toInt();

    if (data == -1)
        return;

    ui->animAnimationSelect->clear();

    // node animations selected
    if (data == 0)
    {
        for (auto& kv : SLScene::current->animManager().animations())
        {
            SLAnimPlayback* play = SLScene::current->animManager().getNodeAnimPlayack(kv.second->name());
            QVariant variant;
            variant.setValue<SLAnimPlayback*>(play);
            ui->animAnimationSelect->addItem(kv.second->name().c_str(), variant);
        }
    }
    // skeleton selected
    else
    {
        int skeletonIndex = data - 1;
        SLSkeleton* skeleton = SLScene::current->animManager().skeletons()[skeletonIndex];
        
        for (auto& kv : skeleton->animations())
        {
            SLAnimPlayback* play = skeleton->getAnimPlayback(kv.second->name());
            QVariant variant;
            variant.setValue<SLAnimPlayback*>(play);
            ui->animAnimationSelect->addItem(kv.second->name().c_str(), variant);
        }
    }

    ui->animAnimationSelect->setCurrentIndex(0);
}
void qtMainWindow::on_animAnimationSelect_currentIndexChanged(int index)
{
    SLAnimPlayback* play = ui->animAnimationSelect->itemData(index).value<SLAnimPlayback*>();
    if (!play) return;
    _selectedAnim = play;

    ui->animSpeedInput->setValue(play->playbackRate());
    ui->animWeightInput->setValue(play->weight());
    ui->animEasingSelect->setCurrentIndex(play->easing());
    ui->animLoopingSelect->setCurrentIndex(play->loop());
    ui->animTimelineSlider->setAnimDuration(play->parentAnimation()->lengthSec());
    ui->animDurationLabel->setText(ui->animTimelineSlider->getDurationTimeString());
}
void qtMainWindow::on_animSkipStartButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->skipToStart();
    updateAllGLWidgets();
}
void qtMainWindow::on_animSkipEndButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->skipToEnd();
    updateAllGLWidgets();
}
void qtMainWindow::on_animPrevKeyframeButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->skipToPrevKeyframe();
    updateAllGLWidgets();
}
void qtMainWindow::on_animNextKeyframeButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->skipToNextKeyframe();
    updateAllGLWidgets();
}
void qtMainWindow::on_animPlayForwardButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->playForward();
    updateAllGLWidgets();
}
void qtMainWindow::on_animPlayBackwardButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->playBackward();
    updateAllGLWidgets();
}
void qtMainWindow::on_animPauseButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->pause();
    updateAllGLWidgets();
}
void qtMainWindow::on_animStopButton_clicked()
{
    if (!_selectedAnim)
        return;

    _selectedAnim->enabled(false);
}
void qtMainWindow::on_animEasingSelect_currentIndexChanged(int index)
{
    if (!_selectedAnim)
        return;
    // @todo add the time preservation back in when the inverse functions are implemented
    SLfloat localTime = _selectedAnim->localTime(); // preserve the local time before switching the easing
    _selectedAnim->easing((SLEasingCurve)index);
    _selectedAnim->localTime(localTime);
}
void qtMainWindow::on_animLoopingSelect_currentIndexChanged(int index)
{
    if (!_selectedAnim)
        return;

    _selectedAnim->loop((SLAnimLooping)(index));
    updateAllGLWidgets();
}
void qtMainWindow::on_animTimelineSlider_valueChanged(int value)
{
    if (!_selectedAnim)
        return;
    
    _selectedAnim->localTime(ui->animTimelineSlider->getNormalizedValue() * _selectedAnim->parentAnimation()->lengthSec());
    updateAllGLWidgets();
}
void qtMainWindow::on_animWeightInput_valueChanged(double d)
{
    if (!_selectedAnim)
        return;

    _selectedAnim->weight(d);
    updateAllGLWidgets();
}
void qtMainWindow::on_animSpeedInput_valueChanged(double d)
{
    if (!_selectedAnim)
        return;

    _selectedAnim->playbackRate(d);
    updateAllGLWidgets();
}
