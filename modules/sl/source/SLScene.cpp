//#############################################################################
//   File:      SLScene.cpp
//   Date:      July 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLScene.h>
#include <Utils.h>
#include <SLKeyframeCamera.h>
#include <SLGLProgramManager.h>
#include <SLSkybox.h>
#include <GlobalTimer.h>
#include <Profiler.h>
#include <SLEntities.h>

//-----------------------------------------------------------------------------
// Global static instances
SLMaterialDefaultGray*           SLMaterialDefaultGray::_instance           = nullptr;
SLMaterialDefaultColorAttribute* SLMaterialDefaultColorAttribute::_instance = nullptr;
#ifdef SL_USE_ENTITIES
SLEntities SLScene::entities;
#endif
//-----------------------------------------------------------------------------
/*! The constructor of the scene.
There will be only one scene for an application and it gets constructed in
the C-interface function slCreateScene in SLInterface.cpp that is called by the
platform and UI-toolkit dependent window initialization.
As examples you can see it in:
  - app_demo_slproject/glfw: glfwMain.cpp in function main()
  - app-Demo-SLProject/android: Java_ch_fhnw_comgRT_glES2Lib_onInit()
  - app_demo_slproject/ios: ViewController.m in method viewDidLoad()
  - _old/app-Demo-Qt: qtGLWidget::initializeGL()
  - _old/app-Viewer-Qt: qtGLWidget::initializeGL()
*/
SLScene::SLScene(const SLstring& name,
                 cbOnSceneLoad   onSceneLoadCallback)
  : SLObject(name),
    _loadTimeMS(0.0f),
    _frameTimesMS(60, 0.0f),
    _updateTimesMS(60, 0.0f),
    _updateAABBTimesMS(60, 0.0f),
    _updateAnimTimesMS(60, 0.0f),
    _updateDODTimesMS(60, 0.0f)
{
    onLoad = onSceneLoadCallback;

    _assetManager     = nullptr;
    _root3D           = nullptr;
    _root2D           = nullptr;
    _skybox           = nullptr;
    _info             = "";
    _stopAnimations   = false;
    _fps              = 0;
    _frameTimeMS      = 0;
    _lastUpdateTimeMS = 0;
}
//-----------------------------------------------------------------------------
/*! The destructor does the final total deallocation of all global resources.
The destructor is called in slTerminate.
*/
SLScene::~SLScene()
{
    unInit();

    // delete global SLGLState instance
    SLGLState::deleteInstance();

    SL_LOG("Destructor      : ~SLScene");
    SL_LOG("------------------------------------------------------------------");
}
//-----------------------------------------------------------------------------
/*! The scene init is called before a new scene is assembled.
 */
void SLScene::init(SLAssetManager* am)
{
    assert(am && "No asset manager passed to scene");

    unInit();

    _assetManager = am;

    // reset all states
    SLGLState::instance()->initAll();

    // reset global light settings
    SLLight::gamma = 1.0f;
    SLLight::globalAmbient.set(0.15f, 0.15f, 0.15f, 1.0f);

    // Reset timing variables
    _frameTimesMS.init(20, 0.0f);
    _updateTimesMS.init(60, 0.0f);
    _updateAnimTimesMS.init(60, 0.0f);
    _updateAABBTimesMS.init(60, 0.0f);
    _updateDODTimesMS.init(60, 0.0f);
}
//-----------------------------------------------------------------------------
/*! The scene uninitializing clears the scenegraph (_root3D) and all global
global resources such as materials, textures & custom shaders loaded with the
scene. The standard shaders, the fonts and the 2D-GUI elements remain. They are
destructed at process end.
*/
void SLScene::unInit()
{
    // delete entire scene graph
    delete _root3D;
    _root3D = nullptr;
    delete _root2D;
    _root2D = nullptr;
    _skybox = nullptr;

    // clear light pointers
    _lights.clear();

    _eventHandlers.clear();
    _animManager.clear();

    _selectedMeshes.clear();
    _selectedNodes.clear();

    // Delete the default material that are scene dependent
    SLMaterialDefaultGray::deleteInstance();
    SLMaterialDefaultColorAttribute::deleteInstance();
}
//-----------------------------------------------------------------------------
//! Updates animations and AABBs
/*! Updates different updatables in the scene after all views got painted:
\n 1) Calculate frame time
\n 2) Update all animations
\n 3) Update AABBs
\n
@return true if really something got updated
*/
bool SLScene::onUpdate(bool renderTypeIsRT,
                       bool voxelsAreShown)
{
    PROFILE_FUNCTION();

    /////////////////////////////
    // 1) Calculate frame time //
    /////////////////////////////

    // Calculate the elapsed time for the animation
    // todo: If slowdown on idle is enabled the delta time will be wrong!
    _frameTimeMS      = GlobalTimer::timeMS() - _lastUpdateTimeMS;
    _lastUpdateTimeMS = GlobalTimer::timeMS();

    // Calculate the frames per second metric
    _frameTimesMS.set(_frameTimeMS);
    SLfloat averagedFrameTimeMS = _frameTimesMS.average();
    if (averagedFrameTimeMS > 0.001f)
        _fps = 1 / _frameTimesMS.average() * 1000.0f;
    else
        _fps = 0.0f;

    SLfloat startUpdateMS = GlobalTimer::timeMS();

    SLbool sceneHasChanged = false;

    //////////////////////////////
    // 2) Update all animations //
    //////////////////////////////

    SLfloat startAnimUpdateMS = GlobalTimer::timeMS();

    if (_root3D)
        _root3D->updateRec();
    if (_root2D)
        _root2D->updateRec();

    // Update node animations
    sceneHasChanged |= !_stopAnimations && _animManager.update(elapsedTimeSec());

    // Do software skinning on all changed skeletons. Update any out of date acceleration structure for RT or if they're being rendered.
    if (_root3D)
    {
        // we use a lambda to inform nodes that share a mesh that the mesh got updated (so we don't have to transfer the root node)
        sceneHasChanged |= _root3D->updateMeshSkins([&](SLMesh* mesh)
                                                    {
            SLVNode nodes = _root3D->findChildren(mesh, true);
            for (auto* node : nodes)
                node->needAABBUpdate(); });

        if (renderTypeIsRT || voxelsAreShown)
            _root3D->updateMeshAccelStructs();
    }

    _updateAnimTimesMS.set(GlobalTimer::timeMS() - startAnimUpdateMS);

    /////////////////////
    // 3) Update AABBs //
    /////////////////////

    // The updateAABBRec call won't generate any overhead if nothing changed
    SLfloat startAAABBUpdateMS = GlobalTimer::timeMS();
    SLNode::numWMUpdates       = 0;
    if (_root3D)
        _root3D->updateAABBRec(renderTypeIsRT);
    if (_root2D)
        _root2D->updateAABBRec(renderTypeIsRT);
    _updateAABBTimesMS.set(GlobalTimer::timeMS() - startAAABBUpdateMS);

#ifdef SL_USE_ENTITIES
    SLfloat startDODUpdateMS = GlobalTimer::timeMS();
    if (entities.size())
    {
        SLMat4f root;
        entities.updateWMRec(0, root);
    }
    _updateDODTimesMS.set(GlobalTimer::timeMS() - startDODUpdateMS);
#endif

    // Finish total updateRec time
    SLfloat updateTimeMS = GlobalTimer::timeMS() - startUpdateMS;
    _updateTimesMS.set(updateTimeMS);

    // SL_LOG("SLScene::onUpdate");
    return sceneHasChanged;
}
//-----------------------------------------------------------------------------
//! Handles the full mesh selection from double-clicks.
/*!
 There are two different selection modes: Full or partial mesh selection.
 <br>
 The full selection is done by double-clicking a mesh. Multiple meshes can be
 selected with SHIFT-double-clicking. The full selection is handled in
 SLScene::selectNodeMesh. The selected nodes are stored in SLScene::_selectedNodes
 and the fully or partially selected meshes are stored in SLScene::_selectedMeshes.
 The SLNode::isSelected and SLMesh::isSelected show if a node or mesh is
 selected. A node can be selected with or without a mesh. If a mesh is
 selected, its node is always also selected. A node without mesh can only be
 selected in the scenegraph window.
 To avoid a node from selection you can set its drawing bit SL_DB_NOTSELECTABLE.
 You should transform a node or mesh and show the properties of a node or mesh
 if only a single node and single full mesh is selected. To get them call
 SLScene::singleNodeSelected() or SLScene::singleMeshFullSelected().
 <br>
 For partial mesh selection see SLMesh::handleRectangleSelection.
*/
void SLScene::selectNodeMesh(SLNode* nodeToSelect, SLMesh* meshToSelect)
{
    // Case 1: Both are nullptr, so unselect all
    if (!nodeToSelect && !meshToSelect)
    {
        deselectAllNodesAndMeshes();
        return;
    }

    if (nodeToSelect && nodeToSelect->drawBit(SL_DB_NOTSELECTABLE))
    {
        SL_LOG("Node is not selectable: %s", nodeToSelect->name().c_str());
        return;
    }

    // Case 2: mesh without node selected: This is not allowed
    if (!nodeToSelect && meshToSelect)
        SL_EXIT_MSG("SLScene::selectNodeMesh: No node or mesh to select.");

    // Search in _selectedNodes vector
    auto foundNode = find(_selectedNodes.begin(),
                          _selectedNodes.end(),
                          nodeToSelect);

    // Case 3: Node without mesh selected
    if (nodeToSelect && !meshToSelect)
    {
        if (foundNode == _selectedNodes.end())
        {
            nodeToSelect->isSelected(true);
            _selectedNodes.push_back(nodeToSelect);
        }
        return;
    }

    // Search in _selectedMeshes vector
    auto foundMesh = find(_selectedMeshes.begin(),
                          _selectedMeshes.end(),
                          meshToSelect);

    // Case 4: nodeToSelect and meshToSelect are not yet selected: so we select them
    if (foundNode == _selectedNodes.end() && foundMesh == _selectedMeshes.end())
    {
        nodeToSelect->isSelected(true);
        _selectedNodes.push_back(nodeToSelect);
        meshToSelect->isSelected(true);
        meshToSelect->deselectPartialSelection();
        _selectedMeshes.push_back(meshToSelect);
        return;
    }

    // Case 5: nodeToSelect is already selected but not the mesh: So select only the mesh
    if (*foundNode == nodeToSelect && foundMesh == _selectedMeshes.end())
    {
        nodeToSelect->isSelected(true);
        meshToSelect->isSelected(true);
        meshToSelect->deselectPartialSelection();
        _selectedMeshes.push_back(meshToSelect);
        return;
    }

    // Case 6: nodeToSelect is not selected but the mesh is selected (from another node)
    if (foundNode == _selectedNodes.end() && *foundMesh == meshToSelect)
    {
        nodeToSelect->isSelected(true);
        _selectedNodes.push_back(nodeToSelect);
        meshToSelect->isSelected(true);
        meshToSelect->deselectPartialSelection();
        _selectedMeshes.push_back(meshToSelect);
        return;
    }

    // Case 7: Both are already selected, so we unselect them.
    if (nodeToSelect && *foundNode == nodeToSelect && *foundMesh == meshToSelect)
    {
        // Check if other mesh from same node is selected
        bool otherMeshIsSelected = false;

        SLMesh* nm = nodeToSelect->mesh();
        for (auto sm : _selectedMeshes)
        {
            if (nm == sm && nm != meshToSelect)
            {
                otherMeshIsSelected = true;
                goto endLoop;
            }
        }

    endLoop:
        if (!otherMeshIsSelected)
        {
            nodeToSelect->isSelected(false);
            _selectedNodes.erase(foundNode);
        }
        meshToSelect->deselectPartialSelection();
        meshToSelect->isSelected(false);
        _selectedMeshes.erase(foundMesh);
        return;
    }

    SL_EXIT_MSG("SLScene::selectNodeMesh: We should not get here.");
}
//-----------------------------------------------------------------------------
//! Deselects all nodes and its meshes.
void SLScene::deselectAllNodesAndMeshes()
{
    for (auto sn : _selectedNodes)
        sn->isSelected(false);
    _selectedNodes.clear();

    for (auto sm : _selectedMeshes)
    {
        sm->deselectPartialSelection();
        sm->isSelected(false);
    }
    _selectedMeshes.clear();
}
//-----------------------------------------------------------------------------
//! Returns the number of camera nodes in the scene
SLint SLScene::numSceneCameras()
{
    if (!_root3D) return 0;
    deque<SLCamera*> cams = _root3D->findChildren<SLCamera>();
    return (SLint)cams.size();
}
//-----------------------------------------------------------------------------
//! Returns the next camera in the scene if there is one
SLCamera* SLScene::nextCameraInScene(SLCamera* activeSVCam)
{
    if (!_root3D) return nullptr;

    deque<SLCamera*> cams = _root3D->findChildren<SLCamera>();

    if (cams.empty()) return nullptr;
    if (cams.size() == 1) return cams[0];

    SLint activeIndex = 0;
    for (SLulong i = 0; i < cams.size(); ++i)
    {
        if (cams[i] == activeSVCam)
        {
            activeIndex = (SLint)i;
            break;
        }
    }

    // find next camera, that is not of type SLKeyframeCamera
    // and if allowAsActiveCam is deactivated
    do
    {
        activeIndex = activeIndex > cams.size() - 2 ? 0 : ++activeIndex;
    } while (dynamic_cast<SLKeyframeCamera*>(cams[(uint)activeIndex]) &&
             !dynamic_cast<SLKeyframeCamera*>(cams[(uint)activeIndex])->allowAsActiveCam());

    return cams[(uint)activeIndex];
}
//-----------------------------------------------------------------------------
void SLScene::initOculus(SLstring shaderDir)
{
    _oculus = std::make_unique<SLGLOculus>(shaderDir);
    _oculus->init();
}
//-----------------------------------------------------------------------------
