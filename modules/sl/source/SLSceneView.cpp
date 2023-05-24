//#############################################################################
//   File:      SLSceneView.cpp
//   Date:      July 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marc Wacker, Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAnimManager.h>
#include <SLCamera.h>
#include <SLLight.h>
#include <SLSceneView.h>
#include <SLSkybox.h>
#include <SLFileStorage.h>
#include <GlobalTimer.h>
#include <SLInputManager.h>
#include <Profiler.h>

#include <utility>

#ifdef SL_EMSCRIPTEN
#    define STB_IMAGE_WRITE_IMPLEMENTATION
#    include "stb_image_write.h"
#endif

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
    _shadowMapTimesMS(60, 0.0f),
    _cullTimesMS(60, 0.0f),
    _draw3DTimesMS(60, 0.0f),
    _draw2DTimesMS(60, 0.0f),
    _screenCaptureIsRequested(false),
    _screenCaptureWaitFrames(0)
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
@param name Name of the sceneview
@param screenWidth Width of the OpenGL frame buffer.
@param screenHeight Height of the OpenGL frame buffer.
@param onWndUpdateCallback Callback for ray tracing update
@param onSelectNodeMeshCallback Callback on node and mesh selection
@param gui Interface for the external Gui build function
@param configPath Path to the config file
*/
void SLSceneView::init(SLstring       name,
                       SLint          screenWidth,
                       SLint          screenHeight,
                       void*          onWndUpdateCallback,
                       void*          onSelectNodeMeshCallback,
                       SLUiInterface* gui,
                       const string&  configPath)
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
    _doMultiSampling  = true;
    _doFrustumCulling = true;
    _doAlphaSorting   = true;
    _doWaitOnIdle     = true;
    _drawBits.allOff();

    _stats2D.clear();
    _stats3D.clear();

    _scrWdiv2 = _scrW >> 1;
    _scrHdiv2 = _scrH >> 1;
    _scrWdivH = (SLfloat)_scrW / (SLfloat)_scrH;

    _renderType = RT_gl;

    _screenCaptureIsRequested = false;

    if (_gui)
        _gui->init(configPath);

    onStartup();
}
//-----------------------------------------------------------------------------
void SLSceneView::unInit()
{
    _camera     = &_sceneViewCamera;
    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;
    _touchDowns = 0;

    _renderType = RT_gl;

    _doDepthTest      = true;
    _doMultiSampling  = true;
    _doFrustumCulling = true;
    _doAlphaSorting   = true;
    _doWaitOnIdle     = true;
    _drawBits.allOff();

    _stats2D.clear();
    _stats3D.clear();
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onInitialize is called by the window system before the first
rendering. It applies all scene rendering attributes with the according
OpenGL function.
*/
void SLSceneView::initSceneViewCamera(const SLVec3f& dir, SLProjType proj)
{
    _sceneViewCamera.camAnim(CA_turntableYUp);
    _sceneViewCamera.name("SceneView Camera");
    _sceneViewCamera.clipNear(.1f);
    _sceneViewCamera.clipFar(2000.0f);
    _sceneViewCamera.maxSpeed(40);
    _sceneViewCamera.stereoEyeSeparation(_sceneViewCamera.focalDist() / 30.0f);
    _sceneViewCamera.setProjection(this, ET_center);
    _sceneViewCamera.projType(proj);

    // fit scenes bounding box in view frustum
    if (_s && _s->root3D())
    {
        // we want to fit the scenes combined aabb in the view frustum
        SLAABBox* sceneBounds = _s->root3D()->aabb();

        _sceneViewCamera.translation(sceneBounds->centerWS(), TS_world);
        _sceneViewCamera.lookAt(sceneBounds->centerWS() + dir,
                                SLVec3f::AXISY,
                                TS_parent);

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
        SLfloat halfTan = tan(Utils::DEG2RAD * _sceneViewCamera.fovV() * 0.5f);

        // @todo There is still a bug when OSX doesn't pass correct GLWidget size
        // correctly set the camera distance...
        SLfloat ar = _sceneViewCamera.aspect();

        // special case for orthographic cameras
        if (proj == P_monoOrthographic)
        {
            // NOTE, the orthographic camera has the ability to zoom by using the following:
            // tan(SL_DEG2RAD*_fovV*0.5f) * pos.length();

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

    _sceneViewCamera.updateAABBRec(false);
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
                           _viewportRect.height);
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
            _gui->onResize(_viewportRect.width, _viewportRect.height);
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
    PROFILE_FUNCTION();

    postSceneLoad();

    SLGLState* stateGL = SLGLState::instance();

    if (_camera)
        stateGL->onInitialize(_camera->background().colors()[0]);
    else
        stateGL->onInitialize(SLCol4f::GRAY);

    _visibleMaterials2D.clear();
    _visibleMaterials3D.clear();
    _nodesOverdrawn.clear();
    _stats2D.clear();
    _stats3D.clear();

    _raytracer.deleteData();
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
        if (_camera)
            _camera->setProjection(this, ET_center);

        // build axis aligned bounding box hierarchy after init
        clock_t t = clock();
        _s->root3D()->updateAABBRec(true);
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
        _s->root2D()->updateAABBRec(true);

        // Collect node statistics
        _stats2D.clear();
        _s->root2D()->statsRec(_stats2D);
    }

    // Reset timing variables
    _shadowMapTimeMS = 0.0f;
    _cullTimeMS      = 0.0f;
    _draw3DTimeMS    = 0.0f;
    _draw2DTimeMS    = 0.0f;
    _cullTimesMS.init(60, 0.0f);
    _draw3DTimesMS.init(60, 0.0f);
    _draw2DTimesMS.init(60, 0.0f);

    initSceneViewCamera();

    if (_gui)
        _gui->onResize(_viewportRect.width, _viewportRect.height);
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
        if (_s && _camera && _camera->projType() == P_stereoSideBySideD)
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
    PROFILE_FUNCTION();

    // SL_LOG("onPaint: -----------------------------------------------------");

    _shadowMapTimesMS.set(_shadowMapTimeMS);
    _cullTimesMS.set(_cullTimeMS);
    _draw3DTimesMS.set(_draw3DTimeMS);
    _draw2DTimesMS.set(_draw2DTimeMS);

    SLbool sceneHasChanged    = false;
    SLbool viewConsumedEvents = false;

    // Only update scene if sceneview got repainted: This check is necessary if
    // this function is called for multiple SceneViews. In this way we only
    // update the geometric representations if all SceneViews got painted once.
    // (can only happen during raytracing)
    if (_gotPainted && _s)
    {
        _gotPainted = false;

        // Process queued up system events and poll custom input devices
        viewConsumedEvents = _inputManager.pollAndProcessEvents(this);

        // update current scene
        sceneHasChanged = _s->onUpdate((_renderType == RT_rt),
                                       drawBit(SL_DB_VOXELS));
    }

    SLbool camUpdated = false;

    // Init and build GUI for all projections except distorted stereo
    if (_gui && _camera && _camera->projType() != P_stereoSideBySideD)
        _gui->onInitNewFrame(_s, this);

    // Clear NO. of draw calls after UI creation
    SLGLVertexArray::totalDrawCalls          = 0;
    SLGLVertexArray::totalPrimitivesRendered = 0;
    SLShadowMap::drawCalls                   = 0;

    if (_s && _camera)
    { // Render the 3D scenegraph by raytracing, pathtracing or OpenGL
        switch (_renderType)
        {
            case RT_gl: camUpdated = draw3DGL(_s->elapsedTimeMS()); break;
            case RT_rt: camUpdated = draw3DRT(); break;
            case RT_pt: camUpdated = draw3DPT(); break;
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
    if (_s && _camera && _camera->projType() == P_stereoSideBySideD)
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
<b>Render shadow maps</b>:
Renders all shadow maps for lights in SLLight::renderShadowMap
</li>
<li>
<b>Updates the camera</b>:
If the active camera has an animation it gets updated first in SLCamera::camUpdate
</li>
<li>
<b>Clear all buffers</b>:
The color and depth buffer are cleared in this step. If the projection is
the Oculus stereo projection also the framebuffer target is bound.
</li>
<li>
<b>Set viewport</b>:
Depending on the projection we set the camera projection and the view
for the center or left eye.
</li>
<li>
<b>Render background</b>:
 If no skybox is used the background is rendered. This can be the camera image
 if the camera is turned on.
</li>
<li>
<b>Set projection and view</b>:
 Sets the camera projection matrix
</li>
<li>
<b>Frustum culling</b>:
During the cull traversal all materials that are seen in the view frustum get
collected in _visibleMaterials. All nodes with their meshes get collected in
SLMaterial::_nodesVisible3D. These materials and nodes get drawn in draw3DGLAll.
</li>
<li>
<b>Draw skybox</b>:
The skybox is draw as first object with frozen depth buffer.
The skybox is always around the active camera.
</li>
<li>
<b>Draw all visible nodes</b>:
 By calling the SLSceneView::draw3DGL all visible nodes of all visible materials
 get drawn sorted by material and transparency. If a stereo projection is set,
 the scene gets drawn a second time for the right eye.
</li>
<li>
<b>Draw right eye for stereo projections</b>
</li>
</ol>
*/
SLbool SLSceneView::draw3DGL(SLfloat elapsedTimeMS)
{
    PROFILE_FUNCTION();

    SLGLState* stateGL = SLGLState::instance();

    preDraw();

    ///////////////////////////
    // 1. Render shadow maps //
    ///////////////////////////

    SLfloat startMS = GlobalTimer::timeMS();

    // Render shadow map for each light which creates shadows
    for (SLLight* light : _s->lights())
    {
        if (light->createsShadows())
            light->renderShadowMap(this, _s->root3D());
    }

    _shadowMapTimeMS = GlobalTimer::timeMS() - startMS;

    /////////////////////////
    // 2. Do camera update //
    /////////////////////////

    startMS = GlobalTimer::timeMS();

    // Update camera animation separately (smooth transition on key movement)
    // todo: ghm1: this is currently only necessary for walking animation (which is somehow always enabled)
    // A problem is also, that it only updates the current camera. This is maybe not what we want for sensor rotated camera.

    SLbool camUpdated = _camera->camUpdate(this, elapsedTimeMS);

    //////////////////////
    // 3. Clear buffers //
    //////////////////////

    // Render into framebuffer if Oculus stereo projection is used
    if (_camera->projType() == P_stereoSideBySideD)
    {
        _s->oculus()->beginFrame();
        _oculusFB.bindFramebuffer((SLint)(_s->oculus()->resolutionScale() * (SLfloat)_scrW),
                                  (SLint)(_s->oculus()->resolutionScale() * (SLfloat)_scrH));
    }

    // Clear color buffer
    stateGL->clearColor(SLVec4f(0.00001f, 0.00001f, 0.00001f, 1.0f));
    stateGL->clearColorDepthBuffer();

    /////////////////////
    // 4. Set viewport //
    /////////////////////

    if (_camera->projType() > P_monoOrthographic)
        _camera->setViewport(this, ET_left);
    else
        _camera->setViewport(this, ET_center);

    //////////////////////////
    // 5. Render background //
    //////////////////////////

    // Render solid color, gradient or textured background from active camera
    if (!_s->skybox())
        _camera->background().render(_viewportRect.width, _viewportRect.height);

    // Change state (only when changed)
    stateGL->multiSample(_doMultiSampling);
    stateGL->depthTest(_doDepthTest);

    //////////////////////////////
    // 6. Set projection & View //
    //////////////////////////////

    // Set projection
    if (_camera->projType() > P_monoOrthographic)
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
    // 7. Frustum culling //
    ////////////////////////

    // Delete all visible nodes from the last frame
    for (auto* material : _visibleMaterials3D)
        material->nodesVisible3D().clear();

    _visibleMaterials3D.clear();
    _nodesOpaque3D.clear();
    _nodesBlended3D.clear();
    _nodesOverdrawn.clear();
    _stats3D.numNodesOpaque  = 0;
    _stats3D.numNodesBlended = 0;
    _camera->setFrustumPlanes();

    if (_s->root3D())
        _s->root3D()->cull3DRec(this);

    _cullTimeMS = GlobalTimer::timeMS() - startMS;

    ////////////////////
    // 8. Draw skybox //
    ////////////////////

    if (_s->skybox())
        _s->skybox()->drawAroundCamera(this);

    ////////////////////////////
    // 9. Draw all visible nodes
    ////////////////////////////

    startMS = GlobalTimer::timeMS();

    draw3DGLAll();

    ///////////////////////////////////////////////
    // 10. Draw right eye for stereo projections //
    ///////////////////////////////////////////////

    if (_camera->projType() > P_monoOrthographic)
    {
        _camera->setViewport(this, ET_right);

        // Only draw backgrounds for stereo projections in different viewports
        if (!_s->skybox() && _camera->projType() < P_stereoLineByLine)
            _camera->background().render(_viewportRect.width, _viewportRect.height);

        _camera->setProjection(this, ET_right);
        _camera->setView(this, ET_right);
        stateGL->depthTest(true);
        if (_s->skybox())
            _s->skybox()->drawAroundCamera(this);
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
 SLSceneView::draw3DGLAll renders by material sorted to avoid expensive material
 switches on the GPU. During the cull traversal all materials that are seen in
 the view frustum get collected in _visibleMaterials. All nodes with their
 meshes get collected in SLMaterial::_nodesVisible3D. <br>
The 3D rendering has then the following steps:
1) Draw nodes with meshes with opaque materials and all helper lines sorted by material<br>
2) Draw remaining opaque nodes (SLCameras, needs redesign)<br>
3) Draw nodes with meshes with blended materials sorted by material and sorted back to front<br>
4) Draw remaining blended nodes (SLText, needs redesign)<br>
5) Draw helpers in overlay mode (not depth buffered)<br>
6) Draw visualization lines of animation curves<br>
*/
void SLSceneView::draw3DGLAll()
{
    PROFILE_FUNCTION();

    // a) Draw nodes with meshes with opaque materials and all helper lines sorted by material
    for (auto material : _visibleMaterials3D)
    {
        if (!material->hasAlpha())
        {
            draw3DGLNodes(material->nodesVisible3D(), false, false);
            _stats3D.numNodesOpaque += (SLuint)material->nodesVisible3D().size();
        }
        draw3DGLLines(material->nodesVisible3D());
    }

    // b) Draw remaining opaque nodes without meshes
    _stats3D.numNodesOpaque += (SLuint)_nodesOpaque3D.size();
    draw3DGLNodes(_nodesOpaque3D, false, false);
    draw3DGLLines(_nodesOpaque3D);

    // c) Draw nodes with meshes with blended materials sorted by material and sorted back to front
    for (auto material : _visibleMaterials3D)
    {
        if (material->hasAlpha())
        {
            draw3DGLNodes(material->nodesVisible3D(), true, _doAlphaSorting);
            _stats3D.numNodesBlended += (SLuint)material->nodesVisible3D().size();
        }
    }

    // d) Draw remaining blended nodes (SLText, needs redesign)
    _stats3D.numNodesBlended += (SLuint)_nodesBlended3D.size();
    draw3DGLNodes(_nodesBlended3D, true, _doAlphaSorting);

    // e) Draw helpers in overlay mode (not depth buffered)
    for (auto material : _visibleMaterials3D)
        draw3DGLLinesOverlay(material->nodesVisible3D());
    draw3DGLLinesOverlay(_nodesOverdrawn);
    draw3DGLLinesOverlay(_nodesOpaque3D);

    // f) Draw visualization lines of animation curves
    _s->animManager().drawVisuals(this);

    // Turn blending off again for correct anaglyph stereo modes
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
    // PROFILE_FUNCTION();

    if (nodes.empty()) return;

    // For blended nodes we activate OpenGL blending and stop depth buffer updates
    SLGLState* stateGL = SLGLState::instance();
    stateGL->blend(alphaBlended);
    stateGL->depthMask(!alphaBlended);

    // Important and expensive step for blended nodes with alpha meshes
    // Depth sort with lambda function by their view distance
    if (depthSorted)
    {
        std::sort(nodes.begin(), nodes.end(), [](SLNode* a, SLNode* b)
                  {
                      if (!a) return false;
                      if (!b) return true;
                      return a->aabb()->sqrViewDist() > b->aabb()->sqrViewDist(); });
    }

    // draw the shapes directly with their wm transform
    for (auto* node : nodes)
    {
        // Set model matrix as the nodes model to world matrix
        stateGL->modelMatrix = node->updateAndGetWM();

        // Finally, draw the nodes mesh
        node->drawMesh(this);
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
    PROFILE_FUNCTION();

    if (nodes.empty()) return;

    SLGLState* stateGL = SLGLState::instance();
    stateGL->blend(false);
    stateGL->depthMask(true);

    // Set the view transform for drawing in world space
    stateGL->modelMatrix.identity();

    // draw the opaque shapes directly w. their wm transform
    for (auto* node : nodes)
    {
        if (node != _camera)
        {
            // Draw first AABB of the shapes but not the camera
            if ((drawBit(SL_DB_BBOX) || node->drawBit(SL_DB_BBOX)) &&
                !node->isSelected())
            {
                if (node->mesh())
                    node->aabb()->drawWS(SLCol4f::RED);
                else
                    node->aabb()->drawWS(SLCol4f::MAGENTA);
            }

            // Draw AABB for selected shapes
            if (node->isSelected())
            {
                node->aabb()->drawWS(SLCol4f::YELLOW);
            }
        }
    }

    GET_GL_ERROR; // Check if any OGL errors occurred
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLLinesOverlay draws the nodes axis and skeleton joints
as overlay
*/
void SLSceneView::draw3DGLLinesOverlay(SLVNode& nodes)
{
    PROFILE_FUNCTION();

    SLGLState* stateGL = SLGLState::instance();

    // draw the opaque shapes directly w. their wm transform
    for (auto* node : nodes)
    {
        if (node != _camera)
        {
            if (drawBit(SL_DB_AXIS) || node->drawBit(SL_DB_AXIS) ||
                drawBit(SL_DB_SKELETON) || node->drawBit(SL_DB_SKELETON) ||
                node->isSelected())
            {
                // Set the view transform
                stateGL->modelMatrix.identity();
                stateGL->blend(false);     // Turn off blending for overlay
                stateGL->depthMask(true);  // Freeze depth buffer for blending
                stateGL->depthTest(false); // Turn of depth test for overlay

                // Draw axis
                if (drawBit(SL_DB_AXIS) ||
                    node->drawBit(SL_DB_AXIS) ||
                    node->isSelected())
                {
                    node->aabb()->drawAxisWS();
                }

                // Draw skeleton
                if (drawBit(SL_DB_SKELETON) ||
                    node->drawBit(SL_DB_SKELETON))
                {
                    // Draw axis of the skeleton joints and its parent bones
                    const SLAnimSkeleton* skeleton = node->skeleton();
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
            else if (drawBit(SL_DB_BRECT) || node->drawBit(SL_DB_BRECT))
            {
                node->aabb()->calculateRectSS();

                SLMat4f prevProjMat = stateGL->projectionMatrix;
                SLMat4f prevViewMat = stateGL->viewMatrix;
                SLfloat w2          = (SLfloat)_scrWdiv2;
                SLfloat h2          = (SLfloat)_scrHdiv2;
                stateGL->projectionMatrix.ortho(-w2, w2, -h2, h2, 1.0f, -1.0f);
                stateGL->viewport(0, 0, _scrW, _scrH);
                stateGL->viewMatrix.identity();
                stateGL->modelMatrix.identity();
                stateGL->modelMatrix.translate(-w2, h2, 1.0f);
                stateGL->depthMask(false); // Freeze depth buffer for blending
                stateGL->depthTest(false); // Disable depth testing

                node->aabb()->rectSS().drawGL(SLCol4f::GREEN);

                stateGL->depthMask(true); // Freeze depth buffer for blending
                stateGL->depthTest(true); // Disable depth testing
                stateGL->projectionMatrix = prevProjMat;
                stateGL->viewMatrix       = prevViewMat;
            }
            else if (node->drawBit(SL_DB_OVERDRAW))
            {
                if (node->mesh() && node->mesh()->mat())
                {
                    SLMesh* mesh     = node->mesh();
                    bool    hasAlpha = mesh->mat()->hasAlpha();

                    // For blended nodes we activate OpenGL blending and stop depth buffer updates
                    stateGL->blend(hasAlpha);
                    stateGL->depthMask(!hasAlpha);
                    stateGL->depthTest(false); // Turn of depth test for overlay

                    // Set model & view transform
                    stateGL->viewMatrix  = stateGL->viewMatrix;
                    stateGL->modelMatrix = node->updateAndGetWM();

                    // Finally, draw the nodes mesh
                    node->drawMesh(this);
                    GET_GL_ERROR; // Check if any OGL errors occurred
                }
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
    PROFILE_FUNCTION();

    SLGLState* stateGL = SLGLState::instance();
    SLfloat    startMS = GlobalTimer::timeMS();

    SLfloat w2               = (SLfloat)_scrWdiv2;
    SLfloat h2               = (SLfloat)_scrHdiv2;
    _stats2D.numNodesOpaque  = 0;
    _stats2D.numNodesBlended = 0;

    // Set orthographic projection with 0,0,0 in the screen center
    if (_s)
    {
        // 1. Set Projection & View
        stateGL->projectionMatrix.ortho(-w2, w2, -h2, h2, 1.0f, -1.0f);
        stateGL->viewport(0, 0, _scrW, _scrH);

        // 2. Pseudo 2D Frustum Culling
        for (auto material : _visibleMaterials2D)
            material->nodesVisible2D().clear();
        _visibleMaterials2D.clear();
        _nodesOpaque2D.clear();
        _nodesBlended2D.clear();
        if (_s->root2D())
            _s->root2D()->cull2DRec(this);

        // 3. Draw all 2D nodes opaque
        draw2DGLNodes();

        // Draw selection rectangle. See also SLMesh::handleRectangleSelection
        if (!_camera->selectRect().isEmpty())
        {
            SLMat4f prevViewMat = stateGL->viewMatrix;
            stateGL->viewMatrix.identity();
            stateGL->modelMatrix.identity();
            stateGL->modelMatrix.translate(-w2, h2, 1.0f);
            stateGL->depthMask(false); // Freeze depth buffer for blending
            stateGL->depthTest(false); // Disable depth testing
            _camera->selectRect().drawGL(SLCol4f::WHITE);
            stateGL->depthMask(true);  // enable depth buffer writing
            stateGL->depthTest(true);  // enable depth testing
            stateGL->viewMatrix = prevViewMat;
        }

        // Draw deselection rectangle. See also SLMesh::handleRectangleSelection
        if (!_camera->deselectRect().isEmpty())
        {
            SLMat4f prevViewMat = stateGL->viewMatrix;
            stateGL->viewMatrix.identity();
            stateGL->modelMatrix.identity();
            stateGL->modelMatrix.translate(-w2, h2, 1.0f);
            stateGL->depthMask(false); // Freeze depth buffer for blending
            stateGL->depthTest(false); // Disable depth testing
            _camera->deselectRect().drawGL(SLCol4f::MAGENTA);
            stateGL->depthMask(true);  // enable depth buffer writing
            stateGL->depthTest(true);  // enable depth testing
            stateGL->viewMatrix = prevViewMat;
        }
    }

    // 4. Draw UI
    if (_gui)
        _gui->onPaint(_viewportRect);

    _draw2DTimeMS = GlobalTimer::timeMS() - startMS;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw2DGLNodes draws 2D nodes from root2D in orthographic projection
*/
void SLSceneView::draw2DGLNodes()
{
    PROFILE_FUNCTION();

    SLfloat    depth   = 1.0f;                           // Render depth between -1 & 1
    SLfloat    cs      = std::min(_scrW, _scrH) * 0.01f; // center size
    SLGLState* stateGL = SLGLState::instance();

    SLMat4f prevViewMat(stateGL->viewMatrix);
    stateGL->viewMatrix.identity();
    stateGL->depthMask(false);   // Freeze depth buffer for blending
    stateGL->depthTest(false);   // Disable depth testing
    stateGL->blend(true);        // Enable blending
    stateGL->polygonLine(false); // Only filled polygons

    // Draw all 2D nodes blended (mostly text font textures)
    // draw the shapes directly with their wm transform
    for (auto material : _visibleMaterials2D)
    {
        _stats2D.numNodesOpaque += (SLuint)material->nodesVisible2D().size();
        for (auto* node : material->nodesVisible2D())
        {
            // Apply world transform
            stateGL->modelMatrix = node->updateAndGetWM();

            // Finally, the nodes meshes
            node->drawMesh(this);
        }
    }

    // Deprecated: SLText node need to be meshes as well
    _stats2D.numNodesOpaque += (SLuint)_nodesBlended2D.size();
    for (auto* node : _nodesBlended2D)
    {
        // Apply world transform
        stateGL->modelMatrix = node->updateAndGetWM();

        // Finally, the nodes meshes
        node->drawMesh(this);
    }

    // Draw rotation helpers during camera animations
    if ((_mouseDownL || _mouseDownM) && _touchDowns == 0)
    {
        if (_camera->camAnim() == CA_turntableYUp ||
            _camera->camAnim() == CA_turntableZUp)
        {
            stateGL->modelMatrix.identity();
            stateGL->modelMatrix.translate(0, 0, depth);

            SLVVec3f centerRombusPoints = {{-cs, 0, 0},
                                           {0, -cs, 0},
                                           {cs, 0, 0},
                                           {0, cs, 0}};
            _vaoTouch.clearAttribs();
            _vaoTouch.generateVertexPos(&centerRombusPoints);
            SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);

            _vaoTouch.drawArrayAsColored(PT_lineLoop, yelloAlpha);
        }
        else if (_camera->camAnim() == CA_trackball)
        {
            stateGL->modelMatrix.identity();
            stateGL->modelMatrix.translate(0, 0, depth);

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
        }
    }

    stateGL->viewMatrix = prevViewMat;
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
    // mouse coordinates are top-left, viewport is bottom-left)
    SLint x = scrX - _viewportRect.x;
    SLint y = scrY - ((_scrH - _viewportRect.height) - _viewportRect.y);

    // Pass the event to imgui
    if (_gui)
    {
        _gui->onMouseDown(button, x, y);

        // Touch devices on iOS or Android have no mouse move event when the
        // finger isn't touching the screen. Therefore, imgui can not detect hovering
        // over an imgui window. Without this extra frame you would have to touch
        // the display twice to open e.g. a menu.
        _gui->renderExtraFrame(_s, this, x, y);

        if (_gui->doNotDispatchMouse())
            return true;
    }

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
            result = _camera->onMouseDown(button, x, y, mod);
        else
            result = true;
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
    // mouse coordinates are top-left, viewport is bottom-left)
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

        // Touch devices on iOS or Android have no mouse move event when the
        // finger isn't touching the screen. Therefore imgui can not detect hovering
        // over an imgui window. Without this extra frame you would have to touch
        // the display twice to open e.g. a menu.
        _gui->renderExtraFrame(_s, this, x, y);
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

            // Update the AABB min & max points in OS
            _s->root3D()->updateAABBRec(true);

            _s->root3D()->hitRec(&pickRay);
            if (pickRay.hitNode)
                cout << "NODE HIT: " << pickRay.hitNode->name() << endl;
        }

        if (pickRay.length < FLT_MAX)
        {

            if (mod & K_shift)
            {
                _s->selectNodeMesh(pickRay.hitNode, pickRay.hitMesh);
                if (onSelectedNodeMesh)
                    onSelectedNodeMesh(pickRay.hitNode, pickRay.hitMesh);
            }
            else
            {
                if (_s->singleNodeSelected() != pickRay.hitNode)
                    _s->deselectAllNodesAndMeshes();

                _s->selectNodeMesh(pickRay.hitNode, pickRay.hitMesh);
                if (onSelectedNodeMesh)
                    onSelectedNodeMesh(pickRay.hitNode, pickRay.hitMesh);
            }
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
/*!
SLSceneView::onTouch2Down gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLSceneView::onTouch2Down(SLint scrX1,
                                 SLint scrY1,
                                 SLint scrX2,
                                 SLint scrY2)
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
SLbool SLSceneView::onTouch2Move(SLint scrX1,
                                 SLint scrY1,
                                 SLint scrX2,
                                 SLint scrY2)
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
SLSceneView::onTouch2Up gets called whenever two fingers lift off a handheld
screen.
*/
SLbool SLSceneView::onTouch2Up(SLint scrX1,
                               SLint scrY1,
                               SLint scrX2,
                               SLint scrY2)
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
    if (key=='L') {doMultiSampling(!doMultiSampling()); return true;}
    if (key=='I') {doWaitOnIdle(!doWaitOnIdle()); return true;}
    if (key=='F') {doFrustumCulling(!doFrustumCulling()); return true;}
    if (key=='J') {doAlphaSorting(!doAlphaSorting()); return true;}
    if (key=='T') {doDepthTest(!doDepthTest()); return true;}
    if (key==K_space) {_s->stopAnimations(!_s->stopAnimations()); return true;}

    if (key=='R') {startRaytracing(5);}
    if (key=='P') {startPathtracing(5, 10);}
#ifdef SL_HAS_OPTIX
    if (key=='R' && mod==K_shift) {startOptixRaytracing(5);}
    if (key=='P' && mod==K_shift) {startOptixPathtracing(5, 100);}
#endif

    if (key=='M') {drawBits()->toggle(SL_DB_MESHWIRED); return true;}
    if (key=='H') {drawBits()->toggle(SL_DB_WITHEDGES); return true;}
    if (key=='O') {drawBits()->toggle(SL_DB_ONLYEDGES); return true;}
    if (key=='N') {drawBits()->toggle(SL_DB_NORMALS); return true;}
    if (key=='B') {drawBits()->toggle(SL_DB_BBOX); return true;}
    if (key=='U') {drawBits()->toggle(SL_DB_BRECT); return true;}
    if (key=='V') {drawBits()->toggle(SL_DB_VOXELS); return true;}
    if (key=='X') {drawBits()->toggle(SL_DB_AXIS); return true;}
    if (key=='C') {drawBits()->toggle(SL_DB_CULLOFF); return true;}
    if (key=='K') {drawBits()->toggle(SL_DB_SKELETON); return true;}

    if (key=='5')
    {   if (_camera->projType() == P_monoPerspective)
            _camera->projType(P_monoOrthographic);
        else _camera->projType(P_monoPerspective);
        if (_renderType == RT_rt && !_raytracer.doContinuous() &&
            _raytracer.state() == rtFinished)
            _raytracer.state(rtReady);
    }

    if (key==K_tab) {switchToNextCameraInScene(); return true;}

    if (key==K_esc)
    {
        if (_camera && _camera->projType() == P_stereoSideBySideD)
            _camera->projType(P_monoPerspective);

        if (!_s->selectedNodes().empty() ||
            !_camera->selectRect().isEmpty() ||
            !_camera->deselectRect().isEmpty())
        {
            _s->deselectAllNodesAndMeshes();
            _camera->selectRect().setZero();
            _camera->deselectRect().setZero();
            return true;
        }

        if(_renderType == RT_rt) _stopRT = true;
        if(_renderType == RT_pt) _stopPT = true;
#ifdef SL_HAS_OPTIX
        if(_renderType == RT_optix_rt) _stopOptixRT = true;
        if(_renderType == RT_optix_pt) _stopOptixPT = true;
#endif
        return true;
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
            _gui->onKeyRelease(key, mod);
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
    SLchar   title[255];
    SLstring profiling = "";

#if PROFILING
    profiling = " *** PROFILING *** ";
#endif

    if (_renderType == RT_rt)
    {
        int numThreads = _raytracer.doDistributed() ? _raytracer.numThreads() : 1;

        if (_raytracer.doContinuous())
        {
            snprintf(title,
                     sizeof(title),
                     "Ray Tracing: %s (fps: %4.1f, Threads: %d)",
                     _s->name().c_str(),
                     _s->fps(),
                     numThreads);
        }
        else
        {
            snprintf(title,
                     sizeof(title),
                     "Ray Tracing: %s (Threads: %d)",
                     _s->name().c_str(),
                     numThreads);
        }
    }
    else if (_renderType == RT_pt)
    {
        snprintf(title,
                 sizeof(title),
                 "Path Tracing: %s (Threads: %d)",
                 _s->name().c_str(),
                 _pathtracer.numThreads());
    }
    else
    {
        string format;
        if (_s->fps() > 5)
            format = "OpenGL Renderer: %s (fps: %4.0f, %u nodes of %u rendered)";
        else
            format = "OpenGL Renderer: %s (fps: %4.1f, %u nodes of %u rendered)";

        snprintf(title,
                 sizeof(title),
                 format.c_str(),
                 _s->name().c_str(),
                 _s->fps(),
                 _stats3D.numNodesOpaque + _stats3D.numNodesBlended,
                 _stats3D.numNodes);
    }
    return profiling + SLstring(title) + profiling;
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
            // s->root3D()->needUpdate();

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
    _raytracer.renderImage(true);

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
            // s->root3D()->needUpdate();

            // Do software skinning on all changed skeletons
            _s->root3D()->updateMeshAccelStructs();
        }

        // Start raytracing
        _pathtracer.render(this);
    }

    // Refresh the render image during PT
    _pathtracer.renderImage(true);

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
    _optixRaytracer.setupScene(this, s()->assetManager());
}
//-----------------------------------------------------------------------------
SLbool SLSceneView::draw3DOptixRT()
{
    SLbool updated = false;

    // if the raytracer not yet got started
    if (_optixRaytracer.state() == rtReady)
    {
        s()->root3D()->needUpdate();

        _optixRaytracer.updateScene(this);

        if (_optixRaytracer.doDistributed())
            _optixRaytracer.renderDistrib();
        else
            _optixRaytracer.renderClassic();
    }

    // Refresh the render image during RT
    _optixRaytracer.renderImage(false);

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
    _optixPathtracer.setupScene(this, s()->assetManager());
}
//-----------------------------------------------------------------------------
SLbool SLSceneView::draw3DOptixPT()
{
    SLbool updated = false;

    // if the path tracer not yet got started
    if (_optixPathtracer.state() == rtReady)
    {
        s()->root3D()->needUpdate();

        // Start path tracing
        _optixPathtracer.updateScene(this);
        _optixPathtracer.render();
    }

    // Refresh the render image during RT
    _optixPathtracer.renderImage(false);

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
//! Saves after n wait frames the front frame buffer as a PNG image.
/* Due to the fact that ImGui needs several frame the render its UI we have to
 * wait a few frames until we can be sure that the executing menu command has
 * disappeared before we can save the screen.
 */
void SLSceneView::saveFrameBufferAsImage(SLstring pathFilename,
                                         cv::Size targetSize)
{
    if (_screenCaptureWaitFrames == 0)
    {
        SLint fbW = _viewportRect.width;
        SLint fbH = _viewportRect.height;

#ifndef SL_EMSCRIPTEN
        GLsizei nrChannels = 3;
#else
        GLsizei nrChannels = 4;
#endif

        GLsizei stride = nrChannels * fbW;
        stride += (stride % 4) ? (4 - stride % 4) : 0;
        GLsizei       bufferSize = stride * fbH;
        vector<uchar> buffer(bufferSize);

        SLGLState::instance()->readPixels(buffer.data());

#ifndef SL_EMSCRIPTEN
        CVMat rgbImg = CVMat(fbH, fbW, CV_8UC3, (void*)buffer.data(), stride);
        cv::cvtColor(rgbImg, rgbImg, cv::COLOR_BGR2RGB);
#else
        CVMat   rgbImg     = CVMat(fbH,
                             fbW,
                             CV_8UC4,
                             (void*)buffer.data(),
                             stride);
        cv::cvtColor(rgbImg, rgbImg, cv::COLOR_RGBA2RGB);
        nrChannels  = 3;
        stride      = nrChannels * fbW;
#endif

        cv::flip(rgbImg, rgbImg, 0);
        if (targetSize.width > 0 && targetSize.height > 0)
            cv::resize(rgbImg, rgbImg, targetSize);

#ifndef SL_EMSCRIPTEN
        vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(6);

        try
        {
            imwrite(pathFilename, rgbImg, compression_params);
            string msg = "Screenshot saved to: " + pathFilename;
            SL_LOG(msg.c_str());
        }
        catch (std::runtime_error& ex)
        {
            string msg = "SLSceneView::saveFrameBufferAsImage: Exception: ";
            msg += ex.what();
            Utils::exitMsg("SLProject", msg.c_str(), __LINE__, __FILE__);
        }
#else
        auto writer = [](void* context, void* data, int size)
        {
            SLIOStream* stream = (SLIOStream*)context;
            stream->write(data, size);
        };

        SLIOStream* stream = SLFileStorage::open(pathFilename,
                                                 IOK_image,
                                                 IOM_write);
        stbi_write_png_to_func(writer,
                               (void*)stream,
                               fbW,
                               fbH,
                               nrChannels,
                               rgbImg.data,
                               stride);
        SLFileStorage::close(stream);
#endif

#if !defined(SL_OS_ANDROID) && !defined(SL_OS_MACIOS) && !defined(SL_EMSCRIPTEN)
        _gui->drawMouseCursor(true);
#endif
        _screenCaptureIsRequested = false;
    }
    else
        _screenCaptureWaitFrames--;
}
//-----------------------------------------------------------------------------
