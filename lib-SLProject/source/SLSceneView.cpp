//#############################################################################
//  File:      SLSceneView.cpp
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLApplication.h>
#include <SLAnimManager.h>
#include <SLCamera.h>
#include <SLLight.h>
#include <SLLightRect.h>
#include <SLSceneView.h>
#include <GlobalTimer.h>
#include <SLInputManager.h>

#include <utility>

//-----------------------------------------------------------------------------
// Milliseconds duration of a long touch event
const SLint SLSceneView::LONGTOUCH_MS = 500;
//-----------------------------------------------------------------------------
//! SLSceneView default constructor
/*! The default constructor adds the this pointer to the sceneView vector in 
SLScene. If an in between element in the vector is zero (from previous sceneviews) 
it will be replaced. The sceneviews _index is the index in the sceneview vector.
It never changes throughout the life of a sceneview. 
*/
SLSceneView::SLSceneView(SLScene* s, int dpi, SLInputManager& inputManager)
  : SLObject(),
    _s(s),
    _dpi(dpi),
    _inputManager(inputManager),
    _cullTimesMS(60, 0.0f),
    _draw3DTimesMS(60, 0.0f),
    _draw2DTimesMS(60, 0.0f)
{
}
//-----------------------------------------------------------------------------
SLSceneView::~SLSceneView()
{
    if (_gui)
        _gui->onClose();

    SL_LOG("Destructor      : ~SLSceneView");
}
//-----------------------------------------------------------------------------
/*! SLSceneView::init initializes default values for an empty scene
\param name Name of the sceneview
\param screenWidth Width of the OpenGL frame buffer.
\param screenHeight Height of the OpenGL frame buffer.
\param onWndUpdateCallback Callback for ray tracing update
\param onSelectNodeMeshCallback Callback on node and mesh selection
\param SLUiInterface Interface for the external Gui build function
*/
void SLSceneView::init(SLstring           name,
                       SLint              screenWidth,
                       SLint              screenHeight,
                       void*              onWndUpdateCallback,
                       void*              onSelectNodeMeshCallback,
                       SLUiInterface*     gui,
                       const std::string& configPath)
{
    _gui        = gui;
    _name       = std::move(name);
    _scrW       = screenWidth;
    _scrH       = screenHeight;
    _gotPainted = true;

    // Set default viewport ratio to the same as the screen
    setViewportFromRatio(SLVec2i(0, 0), VA_center, false);

    // The window update callback function is used to refresh the ray tracing
    // image during the rendering process. The ray tracing image is drawn by OpenGL
    // as a texture on a single quad.
    onWndUpdate = (cbOnWndUpdate)onWndUpdateCallback;

    // The on select node callback is called when a node got selected on double
    // click, so that the UI can react on it.
    onSelectedNodeMesh = (cbOnSelectNodeMesh)onSelectNodeMeshCallback;

    _camera = &_sceneViewCamera;

    // enables and modes
    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;
    _touchDowns = 0;

    _doDepthTest      = true;
    _doMultiSampling  = true; // true=OpenGL multisampling is turned on
    _doFrustumCulling = true; // true=enables view frustum culling
    _doWaitOnIdle     = true;
    _drawBits.allOff();

    _stats2D.clear();
    _stats3D.clear();

    _scrWdiv2 = _scrW >> 1;
    _scrHdiv2 = _scrH >> 1;
    _scrWdivH = (SLfloat)_scrW / (SLfloat)_scrH;
    _scr2fbX  = 1.0f;
    _scr2fbY  = 1.0f;

    _renderType = RT_gl;

    _skybox = nullptr;

    if (_gui)
        _gui->init(configPath);

    onStartup();
}
//-----------------------------------------------------------------------------
void SLSceneView::unInit()
{
    _camera = &_sceneViewCamera;
    _skybox = nullptr;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onInitialize is called by the window system before the first 
rendering. It applies all scene rendering attributes with the according 
OpenGL function.
*/
void SLSceneView::initSceneViewCamera(const SLVec3f& dir, SLProjection proj)
{
    _sceneViewCamera.camAnim(CA_turntableYUp);
    _sceneViewCamera.name("SceneView Camera");
    _sceneViewCamera.clipNear(.1f);
    _sceneViewCamera.clipFar(2000.0f);
    _sceneViewCamera.maxSpeed(40);
    _sceneViewCamera.eyeSeparation(_sceneViewCamera.focalDist() / 30.0f);
    _sceneViewCamera.setProjection(this, ET_center);
    _sceneViewCamera.projection(proj);

    // fit scenes bounding box in view frustum
    if (_s && _s->root3D())
    {
        // we want to fit the scenes combined aabb in the view frustum
        SLAABBox* sceneBounds = _s->root3D()->aabb();

        _sceneViewCamera.translation(sceneBounds->centerWS(), TS_world);
        _sceneViewCamera.lookAt(sceneBounds->centerWS() + dir, SLVec3f::AXISY, TS_parent);

        SLfloat minX = sceneBounds->minWS().x;
        SLfloat minY = sceneBounds->minWS().y;
        SLfloat minZ = sceneBounds->minWS().z;
        SLfloat maxX = sceneBounds->maxWS().x;
        SLfloat maxY = sceneBounds->maxWS().y;
        SLfloat maxZ = sceneBounds->maxWS().z;

        // calculate the min and max points in view space
        SLVec4f vsCorners[8];

        vsCorners[0] = SLVec4f(minX, minY, minZ);
        vsCorners[1] = SLVec4f(maxX, minY, minZ);
        vsCorners[2] = SLVec4f(minX, maxY, minZ);
        vsCorners[3] = SLVec4f(maxX, maxY, minZ);
        vsCorners[4] = SLVec4f(minX, minY, maxZ);
        vsCorners[5] = SLVec4f(maxX, minY, maxZ);
        vsCorners[6] = SLVec4f(minX, maxY, maxZ);
        vsCorners[7] = SLVec4f(maxX, maxY, maxZ);

        SLVec3f vsMin(FLT_MAX, FLT_MAX, FLT_MAX);
        SLVec3f vsMax(FLT_MIN, FLT_MIN, FLT_MIN);

        SLMat4f vm = _sceneViewCamera.updateAndGetWMI();

        for (auto& vsCorner : vsCorners)
        {
            vsCorner = vm * vsCorner;

            vsMin.x = std::min(vsMin.x, vsCorner.x);
            vsMin.y = std::min(vsMin.y, vsCorner.y);
            vsMin.z = std::min(vsMin.z, vsCorner.z);

            vsMax.x = std::max(vsMax.x, vsCorner.x);
            vsMax.y = std::max(vsMax.y, vsCorner.y);
            vsMax.z = std::max(vsMax.z, vsCorner.z);
        }

        SLfloat distX   = 0.0f;
        SLfloat distY   = 0.0f;
        SLfloat halfTan = tan(Utils::DEG2RAD * _sceneViewCamera.fov() * 0.5f);

        // @todo There is still a bug when OSX doesn't pass correct GLWidget size
        // correctly set the camera distance...
        SLfloat ar = _sceneViewCamera.aspect();

        // special case for orthographic cameras
        if (proj == P_monoOrthographic)
        {
            // NOTE, the orthographic camera has the ability to zoom by using the following:
            // tan(SL_DEG2RAD*_fov*0.5f) * pos.length();

            distX = vsMax.x / (ar * halfTan);
            distY = vsMax.y / halfTan;
        }
        else
        {
            // for now we treat all other cases as having a single frustum
            distX = (vsMax.x - vsMin.x) * 0.5f / (ar * halfTan);
            distY = (vsMax.y - vsMin.y) * 0.5f / halfTan;

            distX += vsMax.z;
            distY += vsMax.z;
        }

        SLfloat dist = std::max(distX, distY);

        // set focal distance
        _sceneViewCamera.focalDist(dist);
        _sceneViewCamera.translate(SLVec3f(0, 0, dist), TS_object);
    }

    SLGLState::instance()->modelViewMatrix.identity();
    _sceneViewCamera.updateAABBRec();
    _sceneViewCamera.setInitialState();

    // if no camera exists or in VR mode use the sceneViewCamera
    if (_camera == nullptr)
    {
        _camera = &_sceneViewCamera;
    }

    _camera->needUpdate();
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::switchToSceneViewCamera the general idea for this function is
to switch to the editor camera from a scene camera. It could provide
functionality to stay at the position of the previous camera, or to be reset
to the init position etc..
*/
void SLSceneView::switchToSceneViewCamera()
{
    // if we have an active camera, use its position and orientation
    if (_camera)
    {
        SLMat4f currentWM = _camera->updateAndGetWM();
        SLVec3f position  = currentWM.translation();
        SLVec3f forward(-currentWM.m(8), -currentWM.m(9), -currentWM.m(10));
        _sceneViewCamera.translation(position);
        _sceneViewCamera.lookAt(position + forward);
    }

    _camera = &_sceneViewCamera;
}
//-----------------------------------------------------------------------------
//! Sets the active camera to the next in the scene
void SLSceneView::switchToNextCameraInScene()
{
    if (!_s)
        return;

    SLCamera* nextCam = _s->nextCameraInScene(this->camera());

    if (nextCam == nullptr)
        return;

    if (nextCam != _camera)
        _camera = nextCam;
    else
        _camera = &_sceneViewCamera;

    _camera->background().rebuild();
}
//-----------------------------------------------------------------------------
//! Sets the viewport ratio and the viewport rectangle
void SLSceneView::setViewportFromRatio(const SLVec2i&  vpRatio,
                                       SLViewportAlign vpAlign,
                                       SLbool          vpSameAsVideo)
{
    assert(_scrW > 0 && _scrH > 0 && "SLSceneView::setViewportFromRatio: Invalid screen size");

    _viewportRatio       = vpRatio;
    _viewportAlign       = vpAlign;
    _viewportSameAsVideo = vpSameAsVideo;

    // Shortcut if viewport is the same as the screen
    if (vpRatio == SLVec2i::ZERO)
    {
        _viewportRect.set(0, 0, _scrW, _scrH);
        _viewportAlign = VA_center;
        if (_gui)
            _gui->onResize(_viewportRect.width,
                           _viewportRect.height,
                           _scr2fbX,
                           _scr2fbY);
        return;
    }

    // Calculate viewport rect from viewport aspect ratio
    SLfloat vpWdivH = (float)vpRatio.x / (float)vpRatio.y;
    _scrWdivH       = (float)_scrW / (float)_scrH;
    SLRecti vpRect;

    if (_scrWdivH > vpWdivH)
    {
        vpRect.width  = (int)((float)_scrH * vpWdivH);
        vpRect.height = _scrH;
        vpRect.y      = 0;

        switch (vpAlign)
        {
            // viewport coordinates are bottom-left
            case VA_leftOrTop: vpRect.x = 0; break;
            case VA_rightOrBottom: vpRect.x = _scrW - vpRect.width; break;
            case VA_center:
            default: vpRect.x = (_scrW - vpRect.width) / 2; break;
        }
    }
    else
    {
        vpRect.width  = _scrW;
        vpRect.height = (SLint)((float)_scrW / (float)vpWdivH);
        vpRect.x      = 0;

        switch (vpAlign)
        {
            // viewport coordinates are bottom-left
            case VA_leftOrTop: vpRect.y = _scrH - vpRect.height; break;
            case VA_rightOrBottom: vpRect.y = 0; break;
            case VA_center:
            default: vpRect.y = (_scrH - vpRect.height) / 2; break;
        }
    }

    if (SLRecti(_scrW, _scrH).contains(vpRect))
    {
        _viewportRect = vpRect;
        if (_gui)
            _gui->onResize(_viewportRect.width,
                           _viewportRect.height,
                           _scr2fbX,
                           _scr2fbY);
    }
    else
        SL_EXIT_MSG("SLSceneView::viewport: Viewport is bigger than the screen!");
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onInitialize is called by the window system before the first 
rendering. It applies all scene rendering attributes with the according 
OpenGL function.
*/
void SLSceneView::onInitialize()
{
    postSceneLoad();

    SLGLState* stateGL = SLGLState::instance();

    if (_camera)
        stateGL->onInitialize(_camera->background().colors()[0]);
    else
        stateGL->onInitialize(SLCol4f::GRAY);

    _nodesBlended.clear();
    _nodesVisible.clear();
    _nodesOverdrawn.clear();
    _nodesVisible2D.clear();
    _stats2D.clear();
    _stats3D.clear();

    _raytracer.clearData();
    _renderType   = RT_gl;
    _isFirstFrame = true;

#ifdef SL_HAS_OPTIX
    _optixRaytracer.setupOptix();
    _optixPathtracer.setupOptix();
#endif

    // init 3D scene with initial depth 1
    if (_s && _s->root3D() && _s->root3D()->aabb()->radiusOS() < 0.0001f)
    {
        // Init camera so that its frustum is set
        _camera->setProjection(this, ET_center);

        // build axis aligned bounding box hierarchy after init
        clock_t t = clock();
        _s->root3D()->updateAABBRec();
        _s->root3D()->updateMeshAccelStructs();

        SL_LOG("Time for AABBs  : %5.3f sec.",
               (SLfloat)(clock() - t) / (SLfloat)CLOCKS_PER_SEC);

        // Collect node statistics
        _s->root3D()->statsRec(_stats3D);

        // Warn if there are no light in scene
        if (_s->lights().empty())
            SL_LOG("**** No Lights found in scene! ****");
    }

    // init 2D scene with initial depth 1
    if (_s && _s->root2D() && _s->root2D()->aabb()->radiusOS() < 0.0001f)
    {
        // build axis aligned bounding box hierarchy after init
        _s->root2D()->updateAABBRec();

        // Collect node statistics
        _stats2D.clear();
        _s->root2D()->statsRec(_stats2D);
    }

    // Reset timing variables
    _cullTimeMS = 0.0f;
    _draw3DTimeMS = 0.0f;
    _draw2DTimeMS = 0.0f;
    _cullTimesMS.init(60, 0.0f);
    _draw3DTimesMS.init(60, 0.0f);
    _draw2DTimesMS.init(60, 0.0f);

    initSceneViewCamera();

    // init conetracer if possible:
#ifdef GL_VERSION_4_4
    if (gl3wIsSupported(4, 4))
    {
        // The world's bounding box should not change during runtime.
        if (_s && _s->root3D())
            _conetracer.init(_scrW, _scrH, _s->root3D()->aabb()->minWS(), _s->root3D()->aabb()->maxWS());
    }
#endif

    if (_gui)
        _gui->onResize(_viewportRect.width,
                       _viewportRect.height,
                       1.0f,
                       1.0f);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onResize is called by the window system before the first 
rendering and whenever the window changes its size.
*/
void SLSceneView::onResize(SLint width, SLint height)
{
    // On OSX and Qt this can be called with invalid values > so exit
    if (width == 0 || height == 0) return;

    if (_scrW != width || _scrH != height)
    {
        _scrW     = width;
        _scrH     = height;
        _scrWdiv2 = _scrW >> 1; // width / 2
        _scrHdiv2 = _scrH >> 1; // height / 2
        _scrWdivH = (SLfloat)_scrW / (SLfloat)_scrH;

        setViewportFromRatio(_viewportRatio,
                             _viewportAlign,
                             _viewportSameAsVideo);

        // Resize Oculus framebuffer
        if (_s && _camera && _camera->projection() == P_stereoSideBySideD)
        {
            _oculusFB.updateSize((SLint)(_s->oculus()->resolutionScale() * (SLfloat)_viewportRect.width),
                                 (SLint)(_s->oculus()->resolutionScale() * (SLfloat)_viewportRect.height));
            _s->oculus()->renderResolution(_viewportRect.width, _viewportRect.height);
        }

        // Stop raytracing & pathtracing on resize
        if (_renderType != RT_gl)
        {
            _renderType = RT_gl;
            _raytracer.doContinuous(false);
        }
    }
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onPaint is called by window system whenever the window and therefore 
the scene needs to be painted. Depending on the renderer it calls first
SLSceneView::draw3DGL, SLSceneView::draw3DRT or SLSceneView::draw3DPT and
then SLSceneView::draw2DGL for all UI in 2D. The method returns true if either
the 2D or 3D graph was updated or waitEvents is false.
*/
SLbool SLSceneView::onPaint()
{
    _cullTimesMS.set(_cullTimeMS);
    _draw3DTimesMS.set(_draw3DTimeMS);
    _draw2DTimesMS.set(_draw2DTimeMS);

    SLbool sceneHasChanged    = false;
    SLbool viewConsumedEvents = false;
    // Only update scene if sceneview got repainted: This check is necessary if
    // this function is called for multiple SceneViews. In this way we only
    // update the geometric representations if all SceneViews got painted once.
    // (can only happen during raytracing)
    if (_gotPainted)
    {
        _gotPainted = false;
        // Process queued up system events and poll custom input devices
        viewConsumedEvents = _inputManager.pollAndProcessEvents(this);

        //update current scene
        if (_s)
        {
            sceneHasChanged = _s->onUpdate((_renderType == RT_rt),
                                           drawBit(SL_DB_VOXELS));
        }
    }

    SLbool camUpdated = false;

    // Init and build GUI for all projections except distorted stereo
    if (_camera && _camera->projection() != P_stereoSideBySideD)
    {
        if (_gui)
            _gui->onInitNewFrame(_s, this);
    }

    // Clear NO. of draw calls afer UI creation
    SLGLVertexArray::totalDrawCalls = 0;

    if (_s && _camera)
    { // Render the 3D scenegraph by raytracing, pathtracing or OpenGL
        switch (_renderType)
        {
            case RT_gl: camUpdated = draw3DGL(_s->elapsedTimeMS()); break;
            case RT_rt: camUpdated = draw3DRT(); break;
            case RT_pt: camUpdated = draw3DPT(); break;
            case RT_ct: camUpdated = draw3DCT(); break;

#ifdef SL_HAS_OPTIX
            case RT_optix_rt: camUpdated = draw3DOptixRT(); break;
            case RT_optix_pt: camUpdated = draw3DOptixPT(); break;
#endif
        }
    }

    // Render the 2D stuff inclusive the ImGui
    draw2DGL();

    SLGLState::instance()->unbindAnythingAndFlush();

    // Finish Oculus framebuffer
    if (_s && _camera && _camera->projection() == P_stereoSideBySideD)
        _s->oculus()->renderDistortion(_scrW,
                                       _scrH,
                                       _oculusFB.texID(),
                                       _camera->background().colors()[0]);

    // Set gotPainted only to true if RT is not busy
    _gotPainted = _renderType == RT_gl || raytracer()->state() != rtBusy;

    // Return true if it is the first frame or a repaint is needed
    if (_isFirstFrame)
    {
        _isFirstFrame = false;
        return true;
    }

    return !_doWaitOnIdle || camUpdated || sceneHasChanged || viewConsumedEvents;
}
//-----------------------------------------------------------------------------
//! Draws the 3D scene with OpenGL
/*! This is the main routine for updating and drawing the 3D scene for one frame. 
The following steps are processed:
<ol>
<li>
<b>Updates the camera</b>:
If the camera has an animation it gets updated first.
The camera animation is the only animation that is view dependent.
</li>
<li>
<b>Clear Buffers</b>:
The color and depth buffer are cleared in this step. If the projection is
the Oculus stereo projection also the framebuffer target is bound. 
</li>
<li>
<b>Set Projection and View</b>:
Depending on the projection we set the camera projection and the view 
for the center or left eye.
</li>
<li>
<b>Frustum Culling</b>:
The frustum culling traversal fills the vectors SLSceneView::_visibleNodes 
and SLSceneView::_blendNodes with the visible transparent nodes. 
Nodes that are not visible with the current camera are not drawn. 
</li>
<li>
<b>Draw Skybox</b>:
The skybox is draw as first object with frozen depth buffer.
The skybox is allways around the active camera.
</li>
<li>
<b>Draw Opaque and Blended Nodes</b>:
By calling the SLSceneView::draw3D all nodes in the vectors 
SLSceneView::_visibleNodes and SLSceneView::_blendNodes will be drawn.
_blendNodes is a vector with all nodes that contain 1-n meshes with 
alpha material. _visibleNodes is a vector with all visible nodes. 
Even if a node contains alpha meshes it still can contain meshes with 
opaque material. If a stereo projection is set, the scene gets drawn 
a second time for the right eye.
</li>
<li>
<b>Draw Oculus Framebuffer</b>:
If the projection is the Oculus stereo projection the framebuffer image
is drawn.
</li>
</ol>
*/
SLbool SLSceneView::draw3DGL(SLfloat elapsedTimeMS)
{
    SLGLState* stateGL = SLGLState::instance();

    preDraw();

    /////////////////////////
    // 1. Do camera Update //
    /////////////////////////

    SLfloat startMS = GlobalTimer::timeMS();

    // Update camera animation separately (smooth transition on key movement)
    SLbool camUpdated = _camera->camUpdate(elapsedTimeMS);

    //////////////////////
    // 2. Clear Buffers //
    //////////////////////

    // Render into framebuffer if Oculus stereo projection is used
    if (_camera->projection() == P_stereoSideBySideD)
    {
        _s->oculus()->beginFrame();
        _oculusFB.bindFramebuffer((SLint)(_s->oculus()->resolutionScale() * (SLfloat)_scrW),
                                  (SLint)(_s->oculus()->resolutionScale() * (SLfloat)_scrH));
    }

    // Clear color buffer
    stateGL->clearColor(SLCol4f::BLACK);
    stateGL->clearColorDepthBuffer();

    /////////////////////
    // 3. Set viewport //
    /////////////////////

    // Set viewport
    if (_camera->projection() > P_monoOrthographic)
        _camera->setViewport(this, ET_left);
    else
        _camera->setViewport(this, ET_center);

    //////////////////////////
    // 3. Render background //
    //////////////////////////

    // Render solid color, gradient or textured background from active camera
    if (!_skybox)
        _camera->background().render(_viewportRect.width, _viewportRect.height);

    // Change state (only when changed)
    stateGL->multiSample(_doMultiSampling);
    stateGL->depthTest(_doDepthTest);

    //////////////////////////////
    // 4. Set Projection & View //
    //////////////////////////////

    // Set projection
    if (_camera->projection() > P_monoOrthographic)
    {
        _camera->setProjection(this, ET_left);
        _camera->setView(this, ET_left);
    }
    else
    {
        _camera->setProjection(this, ET_center);
        _camera->setView(this, ET_center);
    }

    ////////////////////////
    // 5. Frustum Culling //
    ////////////////////////

    _camera->setFrustumPlanes();
    _nodesBlended.clear();
    _nodesVisible.clear();
    _nodesOverdrawn.clear();
    if (_s->root3D())
        _s->root3D()->cull3DRec(this);
    _cullTimeMS = GlobalTimer::timeMS() - startMS;

    ////////////////////
    // 6. Draw skybox //
    ////////////////////

    if (_skybox)
        _skybox->drawAroundCamera(this);

    ////////////////////////////////////
    // 7. Draw Opaque & Blended Nodes //
    ////////////////////////////////////

    startMS = GlobalTimer::timeMS();

    draw3DGLAll();

    // For stereo draw for right eye
    if (_camera->projection() > P_monoOrthographic)
    {
        _camera->setViewport(this, ET_right);

        // Only draw backrounds for stereo projections in different viewports
        if (!_skybox && _camera->projection() < P_stereoLineByLine)
            _camera->background().render(_viewportRect.width, _viewportRect.height);

        _camera->setProjection(this, ET_right);
        _camera->setView(this, ET_right);
        stateGL->depthTest(true);
        draw3DGLAll();
    }

    // Enable all color channels again
    stateGL->colorMask(1, 1, 1, 1);

    _draw3DTimeMS = GlobalTimer::timeMS() - startMS;

    postDraw();

    GET_GL_ERROR; // Check if any OGL errors occurred
    return camUpdated;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLAll renders the opaque nodes before blended nodes and
the blended nodes have to be drawn from back to front.
During the cull traversal all nodes with alpha materials are flagged and 
added the to the vector _alphaNodes. The _visibleNodes vector contains all
nodes because a node with alpha meshes still can have nodes with opaque
material. To avoid double drawing the SLNode::drawMeshes draws in the blended
pass only the alpha meshes and in the opaque pass only the opaque meshes.
*/
void SLSceneView::draw3DGLAll()
{
    // 1) Draw first the opaque shapes and all helper lines (normals and AABBs)
    draw3DGLNodes(_nodesVisible, false, false);
    draw3DGLLines(_nodesVisible);
    draw3DGLLines(_nodesBlended);

    // 2) Draw blended nodes sorted back to front
    draw3DGLNodes(_nodesBlended, true, true);

    // 3) Draw helper
    draw3DGLLinesOverlay(_nodesVisible);
    draw3DGLLinesOverlay(_nodesBlended);
    draw3DGLLinesOverlay(_nodesOverdrawn);

    // 4) Draw visualization lines of animation curves
    _s->animManager().drawVisuals(this);

    // 5) Turn blending off again for correct anaglyph stereo modes
    SLGLState* stateGL = SLGLState::instance();
    stateGL->blend(false);
    stateGL->depthMask(true);
    stateGL->depthTest(true);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLNodes draws the nodes meshes from the passed node vector
directly with their world transform after the view transform.
*/
void SLSceneView::draw3DGLNodes(SLVNode& nodes,
                                SLbool   alphaBlended,
                                SLbool   depthSorted)
{
    if (nodes.empty()) return;

    // For blended nodes we activate OpenGL blending and stop depth buffer updates
    SLGLState* stateGL = SLGLState::instance();
    stateGL->blend(alphaBlended);
    stateGL->depthMask(!alphaBlended);

    // Important and expensive step for blended nodes with alpha meshes
    // Depth sort with lambda function by their view distance
    if (depthSorted)
    {
        std::sort(nodes.begin(), nodes.end(), [](SLNode* a, SLNode* b) {
            if (!a) return false;
            if (!b) return true;
            return a->aabb()->sqrViewDist() > b->aabb()->sqrViewDist();
        });
    }

    // draw the shapes directly with their wm transform
    for (auto* node : nodes)
    {
        // Set the view transform
        stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);

        // Apply world transform
        stateGL->modelViewMatrix.multiply(node->updateAndGetWM().m());

        // Finally the nodes meshes
        node->drawMeshes(this);
    }

    GET_GL_ERROR; // Check if any OGL errors occurred
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLLines draws the AABB from the passed node vector directly
with their world coordinates after the view transform. The lines must be drawn
without blending.
Colors:
Red   : AABB of nodes with meshes
Pink  : AABB of nodes without meshes (only child nodes)
Yellow: AABB of selected node 
*/
void SLSceneView::draw3DGLLines(SLVNode& nodes)
{
    if (nodes.empty()) return;

    SLGLState* stateGL = SLGLState::instance();
    stateGL->blend(false);
    stateGL->depthMask(true);

    // Set the view transform
    stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);

    // draw the opaque shapes directly w. their wm transform
    for (auto* node : nodes)
    {
        if (node != _camera)
        {
            // Draw first AABB of the shapes but not the camera
            if ((drawBit(SL_DB_BBOX) || node->drawBit(SL_DB_BBOX)) &&
                !node->drawBit(SL_DB_SELECTED))
            {
                if (node->numMeshes() > 0)
                    node->aabb()->drawWS(SLCol3f(1, 0, 0));
                else
                    node->aabb()->drawWS(SLCol3f(1, 0, 1));
            }

            // Draw AABB for selected shapes
            if (node->drawBit(SL_DB_SELECTED))
            {
                if (node == _s->selectedNode() || !_camera->selectedRect().isEmpty())
                    node->aabb()->drawWS(SLCol3f(1, 1, 0));
                else
                {
                    // delete selection bits from previous rectangle selection
                    if (node != _s->selectedNode() && _camera->selectedRect().isEmpty())
                        node->drawBits()->off(SL_DB_SELECTED);
                }
            }
        }
    }

    GET_GL_ERROR; // Check if any OGL errors occurred
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLLinesOverlay draws the nodes axis and skeleton joints
as overlayed
*/
void SLSceneView::draw3DGLLinesOverlay(SLVNode& nodes)
{
    // draw the opaque shapes directly w. their wm transform
    for (auto* node : nodes)
    {
        if (node != _camera)
        {
            if (drawBit(SL_DB_AXIS) || node->drawBit(SL_DB_AXIS) ||
                drawBit(SL_DB_SKELETON) || node->drawBit(SL_DB_SKELETON) ||
                node->drawBit(SL_DB_SELECTED))
            {
                // Set the view transform
                SLGLState* stateGL = SLGLState::instance();
                stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);
                stateGL->blend(false);     // Turn off blending for overlay
                stateGL->depthMask(true);  // Freeze depth buffer for blending
                stateGL->depthTest(false); // Turn of depth test for overlay

                // Draw axis
                if (drawBit(SL_DB_AXIS) ||
                    node->drawBit(SL_DB_AXIS) ||
                    node->drawBit(SL_DB_SELECTED))
                {
                    node->aabb()->drawAxisWS();
                }

                // Draw skeleton
                if (drawBit(SL_DB_SKELETON) ||
                    node->drawBit(SL_DB_SKELETON))
                {
                    // Draw axis of the skeleton joints and its parent bones
                    const SLSkeleton* skeleton = node->skeleton();
                    if (skeleton)
                    {
                        for (auto joint : skeleton->joints())
                        {
                            // Get the node wm & apply the joints wm
                            SLMat4f wm = node->updateAndGetWM();
                            wm *= joint->updateAndGetWM();

                            // Get parent node wm & apply the parent joint wm
                            SLMat4f parentWM;
                            if (joint->parent())
                            {
                                parentWM = node->parent()->updateAndGetWM();
                                parentWM *= joint->parent()->updateAndGetWM();
                                joint->aabb()->updateBoneWS(parentWM, false, wm);
                            }
                            else
                                joint->aabb()->updateBoneWS(parentWM, true, wm);

                            joint->aabb()->drawBoneWS();
                        }
                    }
                }
            }
            else if (node->drawBit(SL_DB_OVERDRAW))
            {
                // For blended nodes we activate OpenGL blending and stop depth buffer updates
                SLGLState* stateGL = SLGLState::instance();
                //stateGL->blend(true);
                stateGL->depthMask(false); // Freeze depth buffer for blending
                stateGL->depthTest(false); // Turn of depth test for overlay

                stateGL->blend(node->aabb()->hasAlpha());
                //stateGL->depthMask(!node->aabb()->hasAlpha());

                // Set the view transform
                stateGL->modelViewMatrix.setMatrix(stateGL->viewMatrix);

                // Apply world transform
                stateGL->modelViewMatrix.multiply(node->updateAndGetWM().m());

                // Finally the nodes meshes
                node->drawMeshes(this);

                GET_GL_ERROR; // Check if any OGL errors occurred
            }
        }
    }

    GET_GL_ERROR; // Check if any OGL errors occurred
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw2DGL draws all 2D stuff in ortho projection. So far no
update is done to the 2D scenegraph.
*/
void SLSceneView::draw2DGL()
{
    SLGLState* stateGL = SLGLState::instance();
    SLfloat    startMS = GlobalTimer::timeMS();

    SLfloat w2 = (SLfloat)_scrWdiv2;
    SLfloat h2 = (SLfloat)_scrHdiv2;

    // Set orthographic projection with 0,0,0 in the screen center
    if (_camera && _camera->projection() != P_stereoSideBySideD)
    {
        if (_s)
        {
            // 1. Set Projection & View
            stateGL->projectionMatrix.ortho(-w2, w2, -h2, h2, 1.0f, -1.0f);
            stateGL->viewport(0,
                              0,
                              (int)(_scrW * _scr2fbX),
                              (int)(_scrH * _scr2fbY));

            // 2. Pseudo 2D Frustum Culling
            _nodesVisible2D.clear();
            if (_s->root2D())
                _s->root2D()->cull2DRec(this);

            // 3. Draw all 2D nodes opaque
            draw2DGLNodes();

            // Draw selection rectangle
            /* The selection rectangle is defined in SLScene::selectRect and gets set and
        drawn in SLCamera::onMouseDown and SLCamera::onMouseMove. If the selectRect is
        not empty the SLScene::selectedNode is null. All vertices that are within the
        selectRect are listed in SLMesh::IS32. The selection evaluation is done during
        drawing in SLMesh::draw and is only valid for the current frame.
        All nodes that have selected vertices have their drawbit SL_DB_SELECTED set. */

            if (!_camera->selectedRect().isEmpty())
            {
                stateGL->pushModelViewMatrix();
                stateGL->modelViewMatrix.identity();
                stateGL->modelViewMatrix.translate(-w2, h2, 1.0f);
                stateGL->depthMask(false); // Freeze depth buffer for blending
                stateGL->depthTest(false); // Disable depth testing
                //stateGL->blendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR); // inverts background
                stateGL->blend(true); // Enable blending
                //stateGL->polygonLine(false);       // Only filled polygons

                _camera->selectedRect().drawGL(SLCol4f::WHITE);

                stateGL->blend(false); // turn off blending
                //stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // std. transparency
                stateGL->depthMask(true); // enable depth buffer writing
                stateGL->depthTest(true); // enable depth testing
                stateGL->popModelViewMatrix();
            }
        }

        // 4. Draw UI
        if (_gui)
        {
            _gui->onPaint(_viewportRect);
        }
    }

    _draw2DTimeMS = GlobalTimer::timeMS() - startMS;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw2DGLNodes draws 2D nodes from root2D in ortho projection.
*/
void SLSceneView::draw2DGLNodes()
{
    SLfloat    depth   = 1.0f;                           // Render depth between -1 & 1
    SLfloat    cs      = std::min(_scrW, _scrH) * 0.01f; // center size
    SLGLState* stateGL = SLGLState::instance();

    stateGL->pushModelViewMatrix();
    stateGL->modelViewMatrix.identity();
    stateGL->depthMask(false);   // Freeze depth buffer for blending
    stateGL->depthTest(false);   // Disable depth testing
    stateGL->blend(true);        // Enable blending
    stateGL->polygonLine(false); // Only filled polygons

    // Draw all 2D nodes blended (mostly text font textures)
    // draw the shapes directly with their wm transform
    for (auto* node : _nodesVisible2D)
    {
        // Apply world transform
        stateGL->modelViewMatrix.multiply(node->updateAndGetWM().m());

        // Finally the nodes meshes
        node->drawMeshes(this);
    }

    // Draw rotation helpers during camera animations
    if ((_mouseDownL || _mouseDownM) && _touchDowns == 0)
    {
        if (_camera->camAnim() == CA_turntableYUp ||
            _camera->camAnim() == CA_turntableZUp)
        {
            stateGL->pushModelViewMatrix();
            stateGL->modelViewMatrix.translate(0, 0, depth);

            SLVVec3f centerRombusPoints = {{-cs, 0, 0},
                                           {0, -cs, 0},
                                           {cs, 0, 0},
                                           {0, cs, 0}};
            _vaoTouch.clearAttribs();
            _vaoTouch.generateVertexPos(&centerRombusPoints);
            SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);

            _vaoTouch.drawArrayAsColored(PT_lineLoop, yelloAlpha);

            stateGL->popModelViewMatrix();
        }
        else if (_camera->camAnim() == CA_trackball)
        {
            stateGL->pushModelViewMatrix();
            stateGL->modelViewMatrix.translate(0, 0, depth);

            // radius = half width or height
            SLfloat r = (SLfloat)(_scrW < _scrH
                                    ? _scrW / 2
                                    : _scrH / 2) *
                        _camera->trackballSize();

            SLVVec3f rombusAndCirclePoints; // = {{-cs,0,0},{0,-cs,0},{cs,0,0},{0,cs,0}};

            // Add points for circle over window
            SLint   circlePoints = 60;
            SLfloat deltaPhi     = Utils::TWOPI / (SLfloat)circlePoints;
            for (SLint i = 0; i < circlePoints; ++i)
            {
                SLVec2f c;
                c.fromPolar(r, i * deltaPhi);
                rombusAndCirclePoints.push_back(SLVec3f(c.x, c.y, 0));
            }
            _vaoTouch.clearAttribs();
            _vaoTouch.generateVertexPos(&rombusAndCirclePoints);
            SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);

            _vaoTouch.drawArrayAsColored(PT_lineLoop, yelloAlpha);

            stateGL->popModelViewMatrix();
        }
    }

    stateGL->popModelViewMatrix();

    stateGL->blend(false);    // turn off blending
    stateGL->depthMask(true); // enable depth buffer writing
    stateGL->depthTest(true); // enable depth testing
    GET_GL_ERROR;             // check if any OGL errors occurred
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseDown gets called whenever a mouse button gets pressed and
dispatches the event to the currently attached event handler object.
*/
SLbool SLSceneView::onMouseDown(SLMouseButton button,
                                SLint         scrX,
                                SLint         scrY,
                                SLKey         mod)
{
    // Correct viewport offset
    // mouse corrds are top-left, viewport is bottom-left)
    SLint x = scrX - _viewportRect.x;
    SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

    // Pass the event to imgui
    if (_gui)
    {
        _gui->onMouseDown(button, x, y);

#ifdef SL_GLES
        // Touch devices on iOS or Android have no mouse move event when the
        // finger isn't touching the screen. Therefore imgui can not detect hovering
        // over an imgui window. Without this extra frame you would have to touch
        // the display twice to open e.g. a menu.
        _gui->renderExtraFrame(_s, this, x, y);
#endif

        if (_gui->doNotDispatchMouse())
            return true;
    }
    //if (ImGui::GetIO().WantCaptureMouse)
    //    return true;

    _mouseDownL = (button == MB_left);
    _mouseDownR = (button == MB_right);
    _mouseDownM = (button == MB_middle);
    _mouseMod   = mod;

    SLbool result = false;
    if (_s && _camera && _s->root3D())
    {
        SLbool eventConsumed = false;
        for (auto* eh : _s->eventHandlers())
        {
            if (eh->onMouseDown(button, x, y, mod))
                eventConsumed = true;
        }

        if (!eventConsumed)
        {
            result = _camera->onMouseDown(button, x, y, mod);
        }
        else
        {
            result = true;
        }
    }

    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseUp gets called whenever a mouse button gets released.
*/
SLbool SLSceneView::onMouseUp(SLMouseButton button,
                              SLint         scrX,
                              SLint         scrY,
                              SLKey         mod)
{
    _touchDowns = 0;

    // Correct viewport offset
    // mouse corrds are top-left, viewport is bottom-left)
    SLint x = scrX - _viewportRect.x;
    SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

    // Continue with ray tracing
    if (_raytracer.state() == rtMoveGL)
    {
        _renderType = RT_rt;
        _raytracer.state(rtReady);
    }

    // Continue with path tracing
    if (_pathtracer.state() == rtMoveGL)
    {
        _renderType = RT_pt;
        _pathtracer.state(rtReady);
    }

    // Pass the event to imgui
    if (_gui)
    {
        _gui->onMouseUp(button, x, y);
        if (_gui->doNotDispatchMouse())
            return true;
    }

    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;

    if (_s && _camera && _s->root3D())
    {
        SLbool result        = false;
        SLbool eventConsumed = false;
        for (auto* eh : _s->eventHandlers())
        {
            if (eh->onMouseUp(button, x, y, mod))
                eventConsumed = true;
        }

        if (!eventConsumed)
        {
            result = _camera->onMouseUp(button, x, y, mod);
        }
        else
        {
            result = true;
        }

        return result;
    }

    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseMove gets called whenever the mouse is moved.
*/
SLbool SLSceneView::onMouseMove(SLint scrX, SLint scrY)
{
    // Correct viewport offset
    // mouse corrds are top-left, viewport is bottom-left)
    SLint x = scrX - _viewportRect.x;
    SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

    // Pass the event to imgui
    if (_gui)
    {
        _gui->onMouseMove(x, y);
        if (_gui->doNotDispatchMouse())
            return true;
    }

    if (!_s || !_s->root3D())
        return false;

    _touchDowns   = 0;
    SLbool result = false;

    SLMouseButton btn;
    if (_mouseDownL)
        btn = MB_left;
    else if (_mouseDownR)
        btn = MB_right;
    else if (_mouseDownM)
        btn = MB_middle;
    else
        btn = MB_none;

    if (_mouseDownL || _mouseDownR || _mouseDownM)
    {
        // Handle move in ray tracing
        if (_renderType == RT_rt && !_raytracer.doContinuous())
        {
            if (_raytracer.state() == rtFinished)
                _raytracer.state(rtMoveGL);
            else
                _raytracer.doContinuous(false);

            _renderType = RT_gl;
        }

        // Handle move in path tracing
        if (_renderType == RT_pt)
        {
            if (_pathtracer.state() == rtFinished)
                _pathtracer.state(rtMoveGL);

            _renderType = RT_gl;
        }
    }

    SLbool eventConsumed = false;
    for (auto* eh : _s->eventHandlers())
    {
        if (eh->onMouseMove(btn, x, y, _mouseMod))
            eventConsumed = true;
    }

    if (!eventConsumed)
        result = _camera->onMouseMove(btn, x, y, _mouseMod);
    else
        result = true;

    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseWheel gets called whenever the mouse wheel is turned.
The parameter wheelPos is an increasing or decreeing counter number.
*/
SLbool SLSceneView::onMouseWheelPos(SLint wheelPos, SLKey mod)
{
    static SLint lastMouseWheelPos = 0;
    SLint        delta             = wheelPos - lastMouseWheelPos;
    lastMouseWheelPos              = wheelPos;
    return onMouseWheel(delta, mod);
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseWheel gets called whenever the mouse wheel is turned.
The parameter delta is positive/negative depending on the wheel direction
*/
SLbool SLSceneView::onMouseWheel(SLint delta, SLKey mod)
{
    // Pass the event to imgui
    if (_gui)
    {
        if (_gui->doNotDispatchMouse())
        {
            _gui->onMouseWheel((SLfloat)delta);
            return true;
        }
    }

    if (!_s || !_s->root3D())
        return false;

    // Handle mouse wheel in RT mode
    if (_renderType == RT_rt && !_raytracer.doContinuous() &&
        _raytracer.state() == rtFinished)
        _raytracer.state(rtReady);

    // Handle mouse wheel in PT mode
    if (_renderType == RT_pt && _pathtracer.state() == rtFinished)
        _pathtracer.state(rtReady);

    SLbool result = _camera->onMouseWheel(delta, mod);

    for (auto* eh : _s->eventHandlers())
    {
        if (eh->onMouseWheel(delta, mod))
            result = true;
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onDoubleClick gets called when a mouse double click or finger 
double tab occurs.
*/
SLbool SLSceneView::onDoubleClick(SLMouseButton button,
                                  SLint         scrX,
                                  SLint         scrY,
                                  SLKey         mod)
{
    if (!_s || !_s->root3D())
        return false;

    // Correct viewport offset
    // mouse coordinates are top-left, viewport is bottom-left)
    SLint x = scrX - _viewportRect.x;
    SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

    SLbool result = false;

    // Do object picking with ray cast
    if (button == MB_left)
    {
        _mouseDownR = false;

        SLRay pickRay(this);
        if (_camera)
        {
            _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);
            _s->root3D()->hitRec(&pickRay);
            if (pickRay.hitNode)
                cout << "NODE HIT: " << pickRay.hitNode->name() << endl;
        }

        if (pickRay.length < FLT_MAX)
        {
            _camera->selectedRect().setZero();
            _s->selectNodeMesh(pickRay.hitNode, pickRay.hitMesh);
            if (onSelectedNodeMesh)
                onSelectedNodeMesh(_s->selectedNode(), _s->selectedMesh());
            result = true;
        }
    }
    else
    {
        result = _camera->onDoubleClick(button, x, y, mod);
        for (auto* eh : _s->eventHandlers())
        {
            if (eh->onDoubleClick(button, x, y, mod))
                result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! SLSceneView::onLongTouch gets called when the mouse or touch is down for
more than 500ms and has not moved.
*/
SLbool SLSceneView::onLongTouch(SLint scrX, SLint scrY)
{
    //SL_LOG("onLongTouch(%d, %d)", x, y);

    // Correct viewport offset
    // mouse coordinates are top-left, viewport is bottom-left)
    SLint x = scrX - _viewportRect.x;
    SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

    return true;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onTouch2Down gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLSceneView::onTouch2Down(SLint scrX1, SLint scrY1, SLint scrX2, SLint scrY2)
{
    if (!_s || !_s->root3D())
        return false;

    // Correct viewport offset
    // mouse coordinates are top-left, viewport is bottom-left)
    SLint x1 = scrX1 - _viewportRect.x;
    SLint y1 = scrY1 - ((_scrH - _viewportRect.height) - _viewportRect.y);
    SLint x2 = scrX2 - _viewportRect.x;
    SLint y2 = scrY2 - ((_scrH - _viewportRect.height) - _viewportRect.y);

    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
    _touchDowns = 2;

    SLbool result = _camera->onTouch2Down(x1, y1, x2, y2);

    for (auto* eh : _s->eventHandlers())
    {
        if (eh->onTouch2Down(x1, y1, x2, y2))
            result = true;
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onTouch2Move gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLSceneView::onTouch2Move(SLint scrX1, SLint scrY1, SLint scrX2, SLint scrY2)
{
    if (!_s || !_s->root3D())
        return false;

    // Correct viewport offset
    SLint x1 = scrX1 - _viewportRect.x;
    SLint y1 = scrY1 - ((_scrH - _viewportRect.height) - _viewportRect.y);
    SLint x2 = scrX2 - _viewportRect.x;
    SLint y2 = scrY2 - ((_scrH - _viewportRect.height) - _viewportRect.y);

    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);

    SLbool result = false;
    if (_touchDowns == 2)
    {
        result = _camera->onTouch2Move(x1, y1, x2, y2);
        for (auto* eh : _s->eventHandlers())
        {
            if (eh->onTouch2Move(x1, y1, x2, y2))
                result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onTouch2Up gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLSceneView::onTouch2Up(SLint scrX1, SLint scrY1, SLint scrX2, SLint scrY2)
{
    if (!_s || !_s->root3D())
        return false;

    // Correct viewport offset
    SLint x1 = scrX1 - _viewportRect.x;
    SLint y1 = scrY1 - ((_scrH - _viewportRect.height) - _viewportRect.y);
    SLint x2 = scrX2 - _viewportRect.x;
    SLint y2 = scrY2 - ((_scrH - _viewportRect.height) - _viewportRect.y);

    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
    _touchDowns = 0;

    SLbool result = _camera->onTouch2Up(x1, y1, x2, y2);
    for (auto* eh : _s->eventHandlers())
    {
        if (eh->onTouch2Up(x1, y1, x2, y2))
            result = true;
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onKeyPress gets get called whenever a key is pressed. Before 
passing the command to the eventhandlers the main key commands are handled by
forwarding them to onCommand. 
*/
SLbool SLSceneView::onKeyPress(SLKey key, SLKey mod)
{
    // Pass the event to imgui
    if (_gui)
    {
        if (_gui->doNotDispatchKeyboard())
        {
            _gui->onKeyPress(key, mod);
            return true;
        }
    }

    if (!_s || !_s->root3D())
        return false;

    // clang-format off
    // We have to coordinate these shortcuts in SLDemoGui::buildMenuBar
    if (key=='M') {doMultiSampling(!doMultiSampling()); return true;}
    if (key=='I') {doWaitOnIdle(!doWaitOnIdle()); return true;}
    if (key=='F') {doFrustumCulling(!doFrustumCulling()); return true;}
    if (key=='T') {doDepthTest(!doDepthTest()); return true;}
    if (key=='O') {_s->stopAnimations(!_s->stopAnimations()); return true;}

    if (key=='G') {renderType(RT_gl); return true;}
    if (key=='R') {startRaytracing(5);}
    if (key=='C') {startConetracing();}

    if (key=='P') {drawBits()->toggle(SL_DB_WIREMESH); return true;}
    if (key=='N') {drawBits()->toggle(SL_DB_NORMALS); return true;}
    if (key=='B') {drawBits()->toggle(SL_DB_BBOX); return true;}
    if (key=='V') {drawBits()->toggle(SL_DB_VOXELS); return true;}
    if (key=='X') {drawBits()->toggle(SL_DB_AXIS); return true;}
    if (key=='C') {drawBits()->toggle(SL_DB_CULLOFF); return true;}
    if (key=='K') {drawBits()->toggle(SL_DB_SKELETON); return true;}

    if (key=='5')
    {   if (_camera->projection() == P_monoPerspective)
             _camera->projection(P_monoOrthographic);
        else _camera->projection(P_monoPerspective);
        if (_renderType == RT_rt && !_raytracer.doContinuous() &&
            _raytracer.state() == rtFinished)
            _raytracer.state(rtReady);
    }

    if (key==K_tab) {switchToNextCameraInScene(); return true;}

    if (key==K_esc && mod==K_ctrl)
    {   if(_renderType == RT_rt)
        {  _stopRT = true;
            return false;
        }
        else if(_renderType == RT_pt)
        {  _stopPT = true;
            return false;
        }
        else return true; // end the program
    }
    // clang-format on

    SLbool result = false;
    if (key || mod)
    {
        // 1) pass it to the camera
        result = _camera->onKeyPress(key, mod);

        // 2) pass it to any other eventhandler
        for (auto* eh : _s->eventHandlers())
        {
            if (eh->onKeyPress(key, mod))
                result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onKeyRelease get called whenever a key is released.
*/
SLbool SLSceneView::onKeyRelease(SLKey key, SLKey mod)
{
    // Pass the event to imgui
    if (_gui)
    {
        if (_gui->doNotDispatchKeyboard())
        {
            _gui->onKeyPress(key, mod);
            return true;
        }
    }

    if (!_s || !_s->root3D())
        return false;

    SLbool result = false;

    if (key || mod)
    {
        // 1) pass it to the camera
        result = _camera->onKeyRelease(key, mod);

        // 2) pass it to any other eventhandler
        for (auto* eh : _s->eventHandlers())
        {
            if (eh->onKeyRelease(key, mod))
                result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onCharInput get called whenever a new charcter comes in
*/
SLbool SLSceneView::onCharInput(SLuint c)
{
    if (_gui)
    {
        if (_gui->doNotDispatchKeyboard())
        {
            _gui->onCharInput(c);
            return true;
        }
    }

    return false;
}
//-----------------------------------------------------------------------------
/*!
Returns the window title with name & FPS
*/
SLstring SLSceneView::windowTitle()
{
    SLchar title[255];

    if (_renderType == RT_rt)
    {
        if (_raytracer.doContinuous())
        {
            sprintf(title,
                    "%s (fps: %4.1f, Threads: %d)",
                    _s->name().c_str(),
                    _s->fps(),
                    _raytracer.numThreads());
        }
        else
        {
            sprintf(title,
                    "%s (Threads: %d)",
                    _s->name().c_str(),
                    _raytracer.numThreads());
        }
    }
    else if (_renderType == RT_pt)
    {
        sprintf(title,
                "%s (Threads: %d)",
                _s->name().c_str(),
                _pathtracer.numThreads());
    }
    else
    {
        SLuint nr = (uint)_nodesVisible.size();
        if (_s->fps() > 5)
            sprintf(title,
                    "%s (fps: %4.0f, %u nodes of %u rendered)",
                    _s->name().c_str(),
                    _s->fps(),
                    nr,
                    _stats3D.numNodes);
        else
            sprintf(title,
                    "%s (fps: %4.1f, %u nodes of %u rendered)",
                    _s->name().c_str(),
                    _s->fps(),
                    nr,
                    _stats3D.numNodes);
    }
    return SLstring(title);
}
//-----------------------------------------------------------------------------
/*!
Starts the ray tracing & sets the RT menu
*/
void SLSceneView::startRaytracing(SLint maxDepth)
{
    _renderType = RT_rt;
    _stopRT     = false;
    _raytracer.maxDepth(maxDepth);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::updateAndRT3D starts the raytracing or refreshes the current RT
image during rendering. The function returns true if an animation was done 
prior to the rendering start.
*/
SLbool SLSceneView::draw3DRT()
{
    SLbool updated = false;

    // if the raytracer not yet got started
    if (_raytracer.state() == rtReady)
    {
        if (_s->root3D())
        {
            // Update transforms and AABBs
            // @Todo: causes multithreading bug in RT
            //s->root3D()->needUpdate();

            // Do software skinning on all changed skeletons
            _s->root3D()->updateMeshAccelStructs();
        }

        // Start raytracing
        if (_raytracer.doDistributed())
            _raytracer.renderDistrib(this);
        else
            _raytracer.renderClassic(this);
    }

    // Refresh the render image during RT
    _raytracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if (_stopRT)
    {
        _renderType = RT_gl;
        updated     = true;
    }

    return updated;
}
//-----------------------------------------------------------------------------
/*!
Starts the pathtracing
*/
void SLSceneView::startPathtracing(SLint maxDepth, SLint samples)
{
    _renderType = RT_pt;
    _stopPT     = false;
    _pathtracer.maxDepth(maxDepth);
    _pathtracer.aaSamples(samples);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::updateAndRT3D starts the raytracing or refreshes the current RT
image during rendering. The function returns true if an animation was done 
prior to the rendering start.
*/
SLbool SLSceneView::draw3DPT()
{
    SLbool updated = false;

    // if the pathtracer not yet got started
    if (_pathtracer.state() == rtReady)
    {
        if (_s->root3D())
        {
            // Update transforms and AABBs
            // @Todo: causes multithreading bug in RT
            //s->root3D()->needUpdate();

            // Do software skinning on all changed skeletons
            _s->root3D()->updateMeshAccelStructs();
        }

        // Start raytracing
        _pathtracer.render(this);
    }

    // Refresh the render image during PT
    _pathtracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if (_stopPT)
    {
        _renderType = RT_gl;
        updated     = true;
    }

    return updated;
}
//-----------------------------------------------------------------------------
#ifdef SL_HAS_OPTIX
void SLSceneView::startOptixRaytracing(SLint maxDepth)
{
    _renderType  = RT_optix_rt;
    _stopOptixRT = false;
    _optixRaytracer.maxDepth(maxDepth);
    _optixRaytracer.setupScene(this);
}
//-----------------------------------------------------------------------------
SLbool SLSceneView::draw3DOptixRT()
{
    SLbool updated = false;

    // if the raytracer not yet got started
    if (_optixRaytracer.state() == rtReady)
    {
        s().root3D()->needUpdate();

        _optixRaytracer.updateScene(this);

        if (_optixRaytracer.doDistributed())
            _optixRaytracer.renderDistrib();
        else
            _optixRaytracer.renderClassic();
    }

    // Refresh the render image during RT
    _optixRaytracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if (_stopOptixRT)
    {
        _renderType = RT_gl;
        updated     = true;
    }

    return updated;
}
//-----------------------------------------------------------------------------
void SLSceneView::startOptixPathtracing(SLint maxDepth, SLint samples)
{
    _renderType  = RT_optix_pt;
    _stopOptixPT = false;
    _optixPathtracer.maxDepth(maxDepth);
    _optixPathtracer.samples(samples);
    _optixPathtracer.setupScene(this);
}
//-----------------------------------------------------------------------------
SLbool SLSceneView::draw3DOptixPT()
{
    SLbool updated = false;

    // if the path tracer not yet got started
    if (_optixPathtracer.state() == rtReady)
    {
        s().root3D()->needUpdate();

        // Start path tracing
        _optixPathtracer.updateScene(this);
        _optixPathtracer.render();
    }

    // Refresh the render image during RT
    _optixPathtracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if (_stopOptixPT)
    {
        _renderType = RT_gl;
        updated     = true;
    }

    return updated;
}
#endif
//-----------------------------------------------------------------------------
/*!
Starts the voxel cone tracing
*/
void SLSceneView::startConetracing()
{
    _renderType = RT_ct;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DCT draws all 3D content with voxel cone tracing.
*/
SLbool SLSceneView::draw3DCT()
{
    //SL_LOG("Rendering VXC ");
    SLfloat startMS = GlobalTimer::timeMS();

    SLbool rendered = _conetracer.render(this);

    _draw3DTimeMS = GlobalTimer::timeMS() - startMS;

    return true;
}
//-----------------------------------------------------------------------------
