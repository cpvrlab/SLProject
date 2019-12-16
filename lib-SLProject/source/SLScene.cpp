//#############################################################################
//  File:      SLScene.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <Utils.h>
#include <SLApplication.h>
#include <SLAssimpImporter.h>
#include <SLInputManager.h>
#include <SLLightDirect.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLText.h>
#include <SLKeyframeCamera.h>

//-----------------------------------------------------------------------------
/*! The constructor of the scene does all one time initialization such as
loading the standard shader programs from which the pointers are stored in
the vector _shaderProgs. Custom shader programs that are loaded in a
scene must be deleted when the scene changes.
The following standard shaders are preloaded:
  - ColorAttribute.vert, Color.frag
  - ColorUniform.vert, Color.frag
  - DiffuseAttribute.vert, Diffuse.frag
  - PerVrtBlinn.vert, PerVrtBlinn.frag
  - PerVrtBlinnTex.vert, PerVrtBlinnTex.frag
  - TextureOnly.vert, TextureOnly.frag
  - PerPixBlinn.vert, PerPixBlinn.frag
  - PerPixBlinnTex.vert, PerPixBlinnTex.frag
  - PerPixCookTorance.vert, PerPixCookTorance.frag
  - PerPixCookToranceTex.vert, PerPixCookToranceTex.frag
  - BumpNormal.vert, BumpNormal.frag
  - BumpNormal.vert, BumpNormalParallax.frag
  - FontTex.vert, FontTex.frag
  - StereoOculus.vert, StereoOculus.frag
  - StereoOculusDistortionMesh.vert, StereoOculusDistortionMesh.frag

There will be only one scene for an application and it gets constructed in
the C-interface function slCreateScene in SLInterface.cpp that is called by the
platform and UI-toolkit dependent window initialization.
As examples you can see it in:
  - app-Demo-SLProject/GLFW: glfwMain.cpp in function main()
  - app-Demo-SLProject/android: Java_ch_fhnw_comgRT_glES2Lib_onInit()
  - app-Demo-SLProject/iOS: ViewController.m in method viewDidLoad()
  - _old/app-Demo-Qt: qtGLWidget::initializeGL()
  - _old/app-Viewer-Qt: qtGLWidget::initializeGL()
*/
SLScene::SLScene(SLstring      name,
                 cbOnSceneLoad onSceneLoadCallback)
  : SLObject(name),
    _frameTimesMS(60, 0.0f),
    _updateTimesMS(60, 0.0f),
    _cullTimesMS(60, 0.0f),
    _draw3DTimesMS(60, 0.0f),
    _draw2DTimesMS(60, 0.0f),
    _updateAABBTimesMS(60, 0.0f),
    _updateAnimTimesMS(60, 0.0f)
{
    SLApplication::scene = this;

    onLoad = onSceneLoadCallback;

    _root3D           = nullptr;
    _root2D           = nullptr;
    _info             = "";
    _selectedMesh     = nullptr;
    _selectedNode     = nullptr;
    _stopAnimations   = false;
    _fps              = 0;
    _frameTimeMS      = 0;
    _lastUpdateTimeMS = 0;

    // Load std. shader programs in order as defined in SLShaderProgs enum in SLenum
    // In the constructor they are added the _shaderProgs vector
    // If you add a new shader here you have to update the SLShaderProgs enum accordingly.
    new SLGLGenericProgram("ColorAttribute.vert", "Color.frag");
    new SLGLGenericProgram("ColorUniform.vert", "Color.frag");
    new SLGLGenericProgram("PerVrtBlinn.vert", "PerVrtBlinn.frag");
    new SLGLGenericProgram("PerVrtBlinnColorAttrib.vert", "PerVrtBlinn.frag");
    new SLGLGenericProgram("PerVrtBlinnTex.vert", "PerVrtBlinnTex.frag");
    new SLGLGenericProgram("TextureOnly.vert", "TextureOnly.frag");
    new SLGLGenericProgram("PerPixBlinn.vert", "PerPixBlinn.frag");
    new SLGLGenericProgram("PerPixBlinnTex.vert", "PerPixBlinnTex.frag");
    new SLGLGenericProgram("PerPixCookTorrance.vert", "PerPixCookTorrance.frag");
    new SLGLGenericProgram("PerPixCookTorranceTex.vert", "PerPixCookTorranceTex.frag");
    new SLGLGenericProgram("BumpNormal.vert", "BumpNormal.frag");
    new SLGLGenericProgram("BumpNormal.vert", "BumpNormalParallax.frag");
    new SLGLGenericProgram("FontTex.vert", "FontTex.frag");
    new SLGLGenericProgram("StereoOculus.vert", "StereoOculus.frag");
    new SLGLGenericProgram("StereoOculusDistortionMesh.vert", "StereoOculusDistortionMesh.frag");

    _numProgsPreload = (SLint)_programs.size();

    // font and video texture are not added to the _textures vector
    SLTexFont::generateFonts();

    _oculus.init();
}
//-----------------------------------------------------------------------------
/*! The destructor does the final total deallocation of all global resources.
The destructor is called in slTerminate.
*/
SLScene::~SLScene()
{
    // Delete all remaining sceneviews
    for (auto sv : _sceneViews)
        delete sv;

    unInit();

    // delete global SLGLState instance
    SLGLState::deleteInstance();

    // clear light pointers
    _lights.clear();

    // delete materials
    for (auto m : _materials) delete m;
    _materials.clear();

    // delete materials
    for (auto m : _meshes) delete m;
    _meshes.clear();

    // delete textures
    for (auto t : _textures) delete t;
    _textures.clear();

    // delete shader programs
    for (auto p : _programs) delete p;
    _programs.clear();

    // delete fonts
    SLTexFont::deleteFonts();

    SL_LOG("Destructor      : ~SLScene\n");
    SL_LOG("------------------------------------------------------------------\n");
}
//-----------------------------------------------------------------------------
/*! The scene init is called before a new scene is assembled.
*/
void SLScene::init()
{
    unInit();

    // reset all states
    SLGLState::instance()->initAll();

    _globalAmbiLight.set(0.2f, 0.2f, 0.2f, 0.0f);
    _selectedNode = nullptr;

    // Reset timing variables
    _frameTimesMS.init(60, 0.0f);
    _updateTimesMS.init(60, 0.0f);
    _cullTimesMS.init(60, 0.0f);
    _draw3DTimesMS.init(60, 0.0f);
    _draw2DTimesMS.init(60, 0.0f);
    _updateAnimTimesMS.init(60, 0.0f);
    _updateAABBTimesMS.init(60, 0.0f);

    // Deactivate in general the device sensors
    SLApplication::devRot.isUsed(false);
    SLApplication::devLoc.isUsed(false);

    _selectedRect.setZero();
}
//-----------------------------------------------------------------------------
/*! The scene uninitializing clears the scenegraph (_root3D) and all global
global resources such as materials, textures & custom shaders loaded with the
scene. The standard shaders, the fonts and the 2D-GUI elements remain. They are
destructed at process end.
*/
void SLScene::unInit()
{
    _selectedMesh = nullptr;
    _selectedNode = nullptr;

    // reset existing sceneviews
    for (auto sv : _sceneViews)
    {
        if (sv != nullptr)
        {
            sv->camera(sv->sceneViewCamera());
            sv->skybox(nullptr);
        }
    }

    // delete entire scene graph
    delete _root3D;
    _root3D = nullptr;
    delete _root2D;
    _root2D = nullptr;

    // clear light pointers
    _lights.clear();

    // delete textures that where allocated during scene construction.
    // The video & raytracing textures are not in this vector and are not dealocated
    for (auto t : _textures)
        delete t;
    _textures.clear();

    // manually clear the default materials (it will get deleted below)
    SLMaterial::defaultGray(nullptr);
    SLMaterial::diffuseAttrib(nullptr);

    // delete materials
    for (auto m : _materials)
        delete m;
    _materials.clear();

    // delete meshes
    for (auto m : _meshes)
        delete m;
    _meshes.clear();

    SLMaterial::current = nullptr;

    // delete custom shader programs but not default shaders
    while (_programs.size() > (SLuint)_numProgsPreload)
    {
        SLGLProgram* sp = _programs.back();
        delete sp;
        _programs.pop_back();
    }

    _eventHandlers.clear();

    _animManager.clear();
}
//-----------------------------------------------------------------------------
//! Processes all queued events and updates animations and AABBs
/*! Updates different updatables in the scene after all views got painted:
\n
\n 1) Calculate frame time
\n 2) Process queued events
\n 3) Update all animations
\n 4) Update AABBs
\n
\return true if really something got updated
*/
bool SLScene::onUpdate()
{
    // Return if not all sceneview got repainted: This check if necessary if
    // this function is called for multiple SceneViews. In this way we only
    // update the geometric representations if all SceneViews got painted once.

    for (auto sv : _sceneViews)
        if (sv != nullptr && !sv->gotPainted())
            return false;

    // Reset all _gotPainted flags
    for (auto sv : _sceneViews)
        if (sv != nullptr)
            sv->gotPainted(false);

    /////////////////////////////
    // 1) Calculate frame time //
    /////////////////////////////

    // Calculate the elapsed time for the animation
    // todo: If slowdown on idle is enabled the delta time will be wrong!
    _frameTimeMS      = SLApplication::timeMS() - _lastUpdateTimeMS;
    _lastUpdateTimeMS = SLApplication::timeMS();

    // Sum up all timings of all sceneviews
    SLfloat sumCullTimeMS   = 0.0f;
    SLfloat sumDraw3DTimeMS = 0.0f;
    SLfloat sumDraw2DTimeMS = 0.0f;
    SLbool  renderTypeIsRT  = false;
    SLbool  voxelsAreShown  = false;
    for (auto sv : _sceneViews)
    {
        if (sv != nullptr)
        {
            sumCullTimeMS += sv->cullTimeMS();
            sumDraw3DTimeMS += sv->draw3DTimeMS();
            sumDraw2DTimeMS += sv->draw2DTimeMS();
            if (!renderTypeIsRT && sv->renderType() == RT_rt)
                renderTypeIsRT = true;
            if (!voxelsAreShown && sv->drawBit(SL_DB_VOXELS))
                voxelsAreShown = true;
        }
    }
    _cullTimesMS.set(sumCullTimeMS);
    _draw3DTimesMS.set(sumDraw3DTimeMS);
    _draw2DTimesMS.set(sumDraw2DTimeMS);

    // Calculate the frames per second metric
    _frameTimesMS.set(_frameTimeMS);
    SLfloat averagedFrameTimeMS = _frameTimesMS.average();
    if (averagedFrameTimeMS > 0.001f)
        _fps = 1 / _frameTimesMS.average() * 1000.0f;
    else
        _fps = 0.0f;

    SLfloat startUpdateMS = SLApplication::timeMS();

    //////////////////////////////
    // 2) Process queued events //
    //////////////////////////////

    // Process queued up system events and poll custom input devices
    SLbool sceneHasChanged = SLApplication::inputManager.pollAndProcessEvents(this);

    //////////////////////////////
    // 3) Update all animations //
    //////////////////////////////

    SLfloat startAnimUpdateMS = SLApplication::timeMS();

    if (_root3D)
        _root3D->update();

    // reset the dirty flag on all skeletons
    for (auto skeleton : _animManager.skeletons())
        skeleton->changed(false);

    sceneHasChanged |= !_stopAnimations && _animManager.update(elapsedTimeSec());

    // Do software skinning on all changed skeletons
    for (auto mesh : _meshes)
    {
        if (mesh->skeleton() && mesh->skeleton()->changed())
        {
            mesh->transformSkin();
            sceneHasChanged = true;
        }

        // update any out of date acceleration structure for RT or if they're being rendered.
        if (renderTypeIsRT || voxelsAreShown)
            mesh->updateAccelStruct();
    }

    _updateAnimTimesMS.set(SLApplication::timeMS() - startAnimUpdateMS);

    /////////////////////
    // 4) Update AABBs //
    /////////////////////

    // The updateAABBRec call won't generate any overhead if nothing changed
    SLfloat startAAABBUpdateMS = SLApplication::timeMS();
    SLNode::numWMUpdates       = 0;
    SLGLState::instance()->modelViewMatrix.identity();
    if (_root3D)
        _root3D->updateAABBRec();
    if (_root2D)
        _root2D->updateAABBRec();
    _updateAABBTimesMS.set(SLApplication::timeMS() - startAAABBUpdateMS);

    // Finish total update time
    SLfloat updateTimeMS = SLApplication::timeMS() - startUpdateMS;
    _updateTimesMS.set(updateTimeMS);

    //SL_LOG("SLScene::onUpdate\n");
    return sceneHasChanged;
}
//-----------------------------------------------------------------------------
//! Sets the _selectedNode to the passed node and flags it as selected
/*! If one node is selected a rectangle selection is reset to zero.
The drawing of the selection is done in SLMesh::draw and SLAABBox::drawWS.
*/
void SLScene::selectNode(SLNode* nodeToSelect)
{
    if (_selectedNode)
        _selectedNode->drawBits()->off(SL_DB_SELECTED);

    if (nodeToSelect)
    {
        if (_selectedNode == nodeToSelect)
        {
            _selectedNode = nullptr;
        }
        else
        {
            _selectedNode = nodeToSelect;
            _selectedNode->drawBits()->on(SL_DB_SELECTED);
        }
        _selectedRect.setZero();
    }
    else
        _selectedNode = nullptr;
    _selectedMesh = nullptr;
}
//-----------------------------------------------------------------------------
//! Sets the _selectedNode and _selectedMesh and flags it as selected
/*! If one node is selected a rectangle selection is reset to zero.
The drawing of the selection is done in SLMesh::draw and SLAABBox::drawWS.
*/
void SLScene::selectNodeMesh(SLNode* nodeToSelect,
                             SLMesh* meshToSelect)
{
    if (_selectedNode)
        _selectedNode->drawBits()->off(SL_DB_SELECTED);

    if (nodeToSelect)
    {
        if (_selectedNode == nodeToSelect && _selectedMesh == meshToSelect)
        {
            _selectedNode = nullptr;
            _selectedMesh = nullptr;
            _selectedRect.setZero();
        }
        else
        {
            _selectedNode = nodeToSelect;
            _selectedMesh = meshToSelect;
            _selectedNode->drawBits()->on(SL_DB_SELECTED);
        }
    }
    else
    {
        _selectedNode = nullptr;
        _selectedMesh = nullptr;
        _selectedRect.setZero();
    }
}
//-----------------------------------------------------------------------------
void SLScene::onLoadAsset(const SLstring& assetFile,
                          SLuint          processFlags)
{
    SLApplication::sceneID = SID_FromFile;

    // Set scene name for new scenes
    if (!_root3D)
        name(Utils::getFileName(assetFile));

    // Try to load assed and add it to the scene root node
    SLAssimpImporter importer;

    ///////////////////////////////////////////////////////////////////////
    SLNode* loaded = importer.load(assetFile, true, nullptr, processFlags);
    ///////////////////////////////////////////////////////////////////////

    // Add root node on empty scene
    if (!_root3D)
    {
        SLNode* scene = new SLNode("Scene");
        _root3D       = scene;
    }

    // Add loaded scene
    if (loaded)
        _root3D->addChild(loaded);

    // Add directional light if no light was in loaded asset
    if (_lights.empty())
    {
        SLAABBox boundingBox = _root3D->updateAABBRec();
        SLfloat  arrowLength = boundingBox.radiusWS() > FLT_EPSILON
                                ? boundingBox.radiusWS() * 0.1f
                                : 0.5f;
        SLLightDirect* light = new SLLightDirect(0, 0, 0, arrowLength, 1.0f, 1.0f, 1.0f);
        SLVec3f        pos   = boundingBox.maxWS().isZero()
                        ? SLVec3f(1, 1, 1)
                        : boundingBox.maxWS() * 1.1f;
        light->translation(pos);
        light->lookAt(pos - SLVec3f(1, 1, 1));
        light->attenuation(1, 0, 0);
        _root3D->addChild(light);
        _root3D->aabb()->reset(); // rest aabb so that it is recalculated
    }

    // call onInitialize on all scene views
    for (auto sv : _sceneViews)
    {
        if (sv != nullptr)
        {
            sv->onInitialize();
        }
    }
}
//-----------------------------------------------------------------------------
//! Returns the number of camera nodes in the scene
SLint SLScene::numSceneCameras()
{
    if (!_root3D) return 0;
    vector<SLCamera*> cams = _root3D->findChildren<SLCamera>();
    return (SLint)cams.size();
}
//-----------------------------------------------------------------------------
//! Returns the next camera in the scene if there is one
SLCamera* SLScene::nextCameraInScene(SLSceneView* activeSV)
{
    if (!_root3D) return nullptr;

    vector<SLCamera*> cams = _root3D->findChildren<SLCamera>();

    if (cams.empty()) return nullptr;
    if (cams.size() == 1) return cams[0];

    SLint activeIndex = 0;
    for (SLulong i = 0; i < cams.size(); ++i)
    {
        if (cams[i] == activeSV->camera())
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
/*! Removes the specified mesh from the meshes resource vector.
*/
bool SLScene::removeMesh(SLMesh* mesh)
{
    assert(mesh);
    for (SLulong i = 0; i < _meshes.size(); ++i)
    {
        if (_meshes[i] == mesh)
        {
            _meshes.erase(_meshes.begin() + i);
            return true;
        }
    }
    return false;
}
//-----------------------------------------------------------------------------
/*! Removes the specified texture from the textures resource vector.
*/
bool SLScene::deleteTexture(SLGLTexture* texture)
{
    assert(texture);
    for (SLulong i = 0; i < _textures.size(); ++i)
    {
        if (_textures[i] == texture)
        {
            delete _textures[i];
            _textures.erase(_textures.begin() + i);
            return true;
        }
    }
    return false;
}
//----------------------------------------------------------------------------
