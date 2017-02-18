//#############################################################################
//  File:      SLSceneView.cpp
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLInterface.h>
#include <SLLight.h>
#include <SLCamera.h>
#include <SLAnimation.h>
#include <SLAnimManager.h>
#include <SLLightSpot.h>
#include <SLLightRect.h>
#include <SLTexFont.h>
#include <SLButton.h>
#include <SLImporter.h>
#include <SLCVCapture.h>

//-----------------------------------------------------------------------------
// Milliseconds duration of a long touch event
const SLint SLSceneView::LONGTOUCH_MS   = 500;
//-----------------------------------------------------------------------------
//! SLSceneView default constructor
/*! The default constructor adds the this pointer to the sceneView vector in 
SLScene. If an in between element in the vector is zero (from previous sceneviews) 
it will be replaced. The sceneviews _index is the index in the sceneview vector.
It never changes throughout the life of a sceneview. 
*/
SLSceneView::SLSceneView() : SLObject()
{ 
    SLScene* s = SLScene::current;
    assert(s && "No SLScene::current instance.");
   
    // Find first a zero pointer gap in
    for (SLint i=0; i<s->sceneViews().size(); ++i)
    {  if (s->sceneViews()[i]==nullptr)
        {   s->sceneViews()[i] = this;
            _index = i;
            return;
        }
    }
   
    // No gaps, so add it and get the index back.
    s->sceneViews().push_back(this);
    _index = (SLuint)s->sceneViews().size() - 1;
}
//-----------------------------------------------------------------------------
SLSceneView::~SLSceneView()
{  
    // Set pointer in SLScene::sceneViews vector to zero but leave it.
    // The remaining sceneviews must keep their index in the vector
    SLScene::current->sceneViews()[_index] = 0;
    SL_LOG("Destructor      : ~SLSceneView\n");
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::init initializes default values for an empty scene
*/
void SLSceneView::init(SLstring name, 
                       SLint screenWidth, 
                       SLint screenHeight,
                       void* onWndUpdateCallback,
                       void* onSelectNodeMeshCallback,
                       void* onShowSystemCursorCallback)
{  
    _name = name;
    _scrW = screenWidth;
    _scrH = screenHeight;
	_vrMode = false;
    _gotPainted = true;

    /* The window update callback function is used to refresh the ray tracing 
    image during the rendering process. The ray tracing image is drawn by OpenGL 
    as a texture on a single quad.*/
    onWndUpdate = (cbOnWndUpdate)onWndUpdateCallback;

    /* The on select node callback is called when a node got selected on double
    click, so that the UI can react on it.*/
    onSelectedNodeMesh = (cbOnSelectNodeMesh)onSelectNodeMeshCallback;

    /* We need access to the system specific cursor and be able to hide it
    if we need to draw our own. 
    @todo could be simplified if we implemented our own SLWindow class */
    onShowSysCursor = (cbOnShowSysCursor)onShowSystemCursorCallback;

    _stateGL = 0;
   
    _camera = 0;
   
    // enables and modes
    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;
    _touchDowns = 0;

    _doDepthTest = true;
    _doMultiSampling = true;    // true=OpenGL multisampling is turned on
    _doFrustumCulling = true;   // true=enables view frustum culling
    _waitEvents = true;
    _usesRotation = false;
    _drawBits.allOff();
       
    _stats.clear();
    _showMenu = true;
    _showStatsTiming = false;
    _showStatsRenderer = false;
    _showStatsMemory = false;
    _showStatsCamera = false;
    _showStatsVideo = false;
    _showInfo = true;
    _showLoading = false;

    _scrWdiv2 = _scrW>>1;
    _scrHdiv2 = _scrH>>1;
    _scrWdivH = (SLfloat)_scrW / (SLfloat)_scrH;
      
    _renderType = RT_gl;

    onStartup(); 
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
    _sceneViewCamera.name("SceneViewCamera");
    _sceneViewCamera.clipNear(.1f);
    _sceneViewCamera.clipFar(2000.0f);
    _sceneViewCamera.maxSpeed(40);
    _sceneViewCamera.eyeSeparation(_sceneViewCamera.focalDist()/30.0f);
    _sceneViewCamera.setProjection(this, ET_center);
  
	// ignore projection if in vr mode
	if(!_vrMode)
		_sceneViewCamera.projection(proj);

    // fit scenes bounding box in view frustum
    SLScene* s = SLScene::current;
    if (s->root3D())
    {
        // we want to fit the scenes combined aabb in the view frustum
        SLAABBox* sceneBounds = s->root3D()->aabb();

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
        
        for(SLint i = 0; i < 8; ++i) 
        {
            vsCorners[i] = vm * vsCorners[i];
            
            vsMin.x = min(vsMin.x, vsCorners[i].x);
            vsMin.y = min(vsMin.y, vsCorners[i].y);
            vsMin.z = min(vsMin.z, vsCorners[i].z);

            vsMax.x = max(vsMax.x, vsCorners[i].x);
            vsMax.y = max(vsMax.y, vsCorners[i].y);
            vsMax.z = max(vsMax.z, vsCorners[i].z);
        }
        
        SLfloat dist = 0.0f;
        SLfloat distX = 0.0f;
        SLfloat distY = 0.0f;
        SLfloat halfTan = tan(SL_DEG2RAD*_sceneViewCamera.fov()*0.5f);

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

        dist = max(distX, distY);

        // set focal distance
        _sceneViewCamera.focalDist(dist);
        _sceneViewCamera.translate(SLVec3f(0, 0, dist), TS_object);
    }

    _stateGL->modelViewMatrix.identity();
    _sceneViewCamera.updateAABBRec();

	// if no camera exists or in VR mode use the sceneViewCamera
	if(_camera == nullptr || _vrMode)
        _camera = &_sceneViewCamera;
	
    _camera->needUpdate();
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::switchToSceneViewCamera the general idea for this function is
to switch to the editor camera from a scene camera. It could provide functionality
to stay at the position of the previous camera, or to be reset to the init position etc..
*/
void SLSceneView::switchToSceneViewCamera()
{
    // if we have an active camera, use its position and orientation for the editor cam
    // @todo This is just placeholder code, doing the stuff below can be done in a much
    //       more elegant way.
    if(_camera) 
    {
        SLMat4f currentWM = _camera->updateAndGetWM();
        SLVec3f position = currentWM.translation();
        SLVec3f forward(-currentWM.m(8), -currentWM.m(9), -currentWM.m(10));
        _sceneViewCamera.translation(position);
        _sceneViewCamera.lookAt(position + forward);
    }
    
    _camera = &_sceneViewCamera;
}

//-----------------------------------------------------------------------------
/*!
SLSceneView::onInitialize is called by the window system before the first 
rendering. It applies all scene rendering attributes with the according 
OpenGL function.
*/
void SLSceneView::onInitialize()
{  
    // loading finished
    showLoading(false);
    postSceneLoad();
    
    SLScene* s = SLScene::current;
    _stateGL = SLGLState::getInstance();
    _stateGL->onInitialize(s->background().colors()[0]);
    
    _blendNodes.clear();
    _visibleNodes.clear();

    _raytracer.clearData();
    _renderType = RT_gl;
    _isFirstFrame = true;

    // init 3D scene with initial depth 1
    if (s->root3D() && s->root3D()->aabb()->radiusOS()==0)
    {
        // Init camera so that its frustum is set
        _camera->setProjection(this, ET_center);

        // build axis aligned bounding box hierarchy after init
        clock_t t = clock();
        s->root3D()->updateAABBRec();

        for (auto mesh : s->meshes())
            mesh->updateAccelStruct();
        
        if (SL::noTestIsRunning())
            SL_LOG("Time for AABBs  : %5.3f sec.\n", 
                   (SLfloat)(clock()-t)/(SLfloat)CLOCKS_PER_SEC);
        
        // Collect node statistics
        _stats.clear();
        s->root3D()->statsRec(_stats);
        if (s->menuGL()) s->menuGL()->statsRec(_stats);
        if (s->menuRT()) s->menuRT()->statsRec(_stats);
        if (s->menuPT()) s->menuPT()->statsRec(_stats);

        // Warn if there are no light in scene
        if (s->lights().size() == 0)
            SL_LOG("\n**** No Lights found in scene! ****\n");
    }

    initSceneViewCamera();
   
    build2DMenus();
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onResize is called by the window system before the first 
rendering and whenever the window changes its size.
*/
void SLSceneView::onResize(SLint width, SLint height)
{  
    SLScene* s = SLScene::current;

    // On OSX and Qt this can be called with invalid values > so exit
    if (width==0 || height==0) return;
   
    if (_scrW!=width || _scrH != height)
    {
        _scrW = width;
        _scrH = height;
        _scrWdiv2 = _scrW>>1;  // width / 2
        _scrHdiv2 = _scrH>>1;  // height / 2
        _scrWdivH = (SLfloat)_scrW/(SLfloat)_scrH;

        //@todo move this code to SLGLOculus (problem with that comes when 
        // using multiple views with different resolutions)
        if (_camera && _camera->projection() == P_stereoSideBySideD)
        {
            _oculusFB.updateSize((SLint)(s->oculus()->resolutionScale()*(SLfloat)_scrW), 
                                 (SLint)(s->oculus()->resolutionScale()*(SLfloat)_scrH));
            s->oculus()->renderResolution(_scrW, _scrH);
        }
      
        // Stop raytracing & pathtracing on resize
        if (_renderType != RT_gl)
        {   _renderType = RT_gl;
            _raytracer.continuous(false);
            s->menu2D(s->menuGL());
            s->menu2D()->hideAndReleaseRec();
            s->menu2D()->drawBits()->off(SL_DB_HIDDEN);
        }
    }
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onPaint is called by window system whenever the window therefore 
the scene needs to be painted. Depending on the renderer it calls first
SLSceneView::draw3DGL, SLSceneView::draw3DRT or SLSceneView::draw3DPT and
then SLSceneView::draw2DGL for all UI in 2D. The method returns true if either
the 2D or 3D graph was updated or waitEvents is false.
*/
SLbool SLSceneView::onPaint()
{  
    SLScene* s = SLScene::current;
    SLGLVertexArray::totalDrawCalls = 0;
    SLbool camUpdated = false;

    // Check time for test scenes
    if (SL::testDurationSec > 0)
        if (testRunIsFinished())
            return false;
    
    if (_camera)
    {   // Render the 3D scenegraph by by raytracing, pathtracing or OpenGL
        switch (_renderType)
        {   case RT_gl: camUpdated = draw3DGL(s->elapsedTimeMS()); break;
            case RT_rt: camUpdated = draw3DRT(); break;
            case RT_pt: camUpdated = draw3DPT(); break;
        }
    };

    // Render the 2D GUI (menu etc.)
    draw2DGL();

    _stateGL->unbindAnythingAndFlush();

    // Finish Oculus framebuffer
    if (_camera && _camera->projection() == P_stereoSideBySideD)
        s->oculus()->endFrame(_scrW, _scrH, _oculusFB.texID());

    // Reset drawcalls
    SLGLVertexArray::totalDrawCalls = 0;

    // Set gotPainted only to true if RT is not busy
    _gotPainted = _renderType==RT_gl || raytracer()->state()!=rtBusy;

    // Return true if it is the first frame or a repaint is needed
    if (_isFirstFrame) 
    {   _isFirstFrame = false;
        return true;
    }

    return !_waitEvents || camUpdated;
}
//-----------------------------------------------------------------------------
//! Draws the 3D scene with OpenGL
/*! This is main routine for updating and drawing the 3D scene for one frame. 
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
    SLScene* s = SLScene::current;

    preDraw();
    
    /////////////////////////
    // 1. Do camera Update //
    /////////////////////////

    SLfloat startMS = s->timeMilliSec();
    
    // Update camera animation separately (smooth transition on key movement)
    SLbool camUpdated = _camera->camUpdate(elapsedTimeMS);

   
    //////////////////////
    // 2. Clear Buffers //
    //////////////////////
    
    // Render into framebuffer if Oculus stereo projection is used
    if (_camera->projection() == P_stereoSideBySideD)
    {   s->oculus()->beginFrame();
        _oculusFB.bindFramebuffer((SLint)(s->oculus()->resolutionScale() * (SLfloat)_scrW), 
                                  (SLint)(s->oculus()->resolutionScale() * (SLfloat)_scrH)); 
    }

    // Clear buffers
    _stateGL->clearColor(s->background().colors()[0]);
    _stateGL->clearColorDepthBuffer();

    // render gradient or textured background
    if (!s->background().isUniform())
        s->background().render(_scrW, _scrH);

    // Change state (only when changed)
    _stateGL->multiSample(_doMultiSampling);
    _stateGL->depthTest(_doDepthTest);
    

    //////////////////////////////
    // 3. Set Projection & View //
    //////////////////////////////
    // Set projection and viewport
    if (_camera->projection() > P_monoOrthographic)
         _camera->setProjection(this, ET_left);
    else _camera->setProjection(this, ET_center);

    // Set view center eye or left eye
    if (_camera->projection() > P_monoOrthographic)
         _camera->setView(this, ET_left);
    else _camera->setView(this, ET_center);

    ////////////////////////
    // 4. Frustum Culling //
    ////////////////////////
   
    _camera->setFrustumPlanes(); 
    _blendNodes.clear();
    _visibleNodes.clear();     
    if (s->root3D())
        s->root3D()->cullRec(this);
   
    _cullTimeMS = s->timeMilliSec() - startMS;


    ////////////////////////////////////
    // 5. Draw Opaque & Blended Nodes //
    ////////////////////////////////////

    startMS = s->timeMilliSec();
    draw3DGLAll();
   
    // For stereo draw for right eye
    if (_camera->projection() > P_monoOrthographic)   
    {   _camera->setProjection(this, ET_right);
        _camera->setView(this, ET_right);
        draw3DGLAll();
    }
      
    // Enable all color channels again
    _stateGL->colorMask(1, 1, 1, 1); 

    _draw3DTimeMS = s->timeMilliSec()-startMS;

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
    draw3DGLNodes(_visibleNodes, false, false);
    draw3DGLLines(_visibleNodes);
    draw3DGLLines(_blendNodes);

    // 2) Draw blended nodes sorted back to front
    draw3DGLNodes(_blendNodes, true, true);

    // 3) Draw helper
    draw3DGLLinesOverlay(_visibleNodes);
    draw3DGLLinesOverlay(_blendNodes);

    // 4) Draw visualization lines of animation curves
    SLScene::current->animManager().drawVisuals(this);

    // 5) Turn blending off again for correct anaglyph stereo modes
    _stateGL->blend(false);
    _stateGL->depthMask(true);
    _stateGL->depthTest(true);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLNodes draws the nodes meshes from the passed node vector
directly with their world transform after the view transform.
*/
void SLSceneView::draw3DGLNodes(SLVNode &nodes,
                                SLbool alphaBlended,
                                SLbool depthSorted)
{
    if (nodes.size() == 0) return;

    // For blended nodes we activate OpenGL blending and stop depth buffer updates
    _stateGL->blend(alphaBlended);
    _stateGL->depthMask(!alphaBlended);

    // Important and expensive step for blended nodes with alpha meshes
    // Depth sort with lambda function by their view distance
    if (depthSorted)
    {   std::sort(nodes.begin(), nodes.end(),
                  [](SLNode* a, SLNode* b)
                  {   if (!a) return false;
                      if (!b) return true;
                      return a->aabb()->sqrViewDist() > b->aabb()->sqrViewDist();
                  });
    }

    // draw the shapes directly with their wm transform
    for(auto node : nodes)
    {
        // Set the view transform
        _stateGL->modelViewMatrix.setMatrix(_stateGL->viewMatrix);

        // Apply world transform
        _stateGL->modelViewMatrix.multiply(node->updateAndGetWM().m());

        // Finally the nodes meshes
        node->drawMeshes(this);
    }

    GET_GL_ERROR;  // Check if any OGL errors occurred
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
void SLSceneView::draw3DGLLines(SLVNode &nodes)
{  
    if (nodes.size() == 0) return;

    _stateGL->blend(false);
    _stateGL->depthMask(true);

    // Set the view transform
    _stateGL->modelViewMatrix.setMatrix(_stateGL->viewMatrix);

    // draw the opaque shapes directly w. their wm transform
    for(auto node : nodes)
    {
        if (node != _camera)
        {
            // Draw first AABB of the shapes but not the camera
            if ((drawBit(SL_DB_BBOX) || node->drawBit(SL_DB_BBOX)) &&
                !node->drawBit(SL_DB_SELECTED))
            {
                if (node->numMeshes() > 0)
                     node->aabb()->drawWS(SLCol3f(1,0,0));
                else node->aabb()->drawWS(SLCol3f(1,0,1));
            }

            // Draw AABB for selected shapes
            if (node->drawBit(SL_DB_SELECTED))
                node->aabb()->drawWS(SLCol3f(1,1,0));
        }
    }
   
    GET_GL_ERROR;        // Check if any OGL errors occurred
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DGLLinesOverlay draws the nodes axis and skeleton joints
as overlayed
*/
void SLSceneView::draw3DGLLinesOverlay(SLVNode &nodes)
{

    // draw the opaque shapes directly w. their wm transform
    for(auto node : nodes)
    {
        if (node != _camera)
        {
            if (drawBit(SL_DB_AXIS) || node->drawBit(SL_DB_AXIS) ||
                drawBit(SL_DB_SKELETON) || node->drawBit(SL_DB_SKELETON))
            {
                // Set the view transform
                _stateGL->modelViewMatrix.setMatrix(_stateGL->viewMatrix);
                _stateGL->blend(false);      // Turn off blending for overlay
                _stateGL->depthMask(true);   // Freeze depth buffer for blending
                _stateGL->depthTest(false);  // Turn of depth test for overlay

                // Draw axis
                if (drawBit(SL_DB_AXIS) || node->drawBit(SL_DB_AXIS))
                    node->aabb()->drawAxisWS();

                // Draw skeleton
                if (drawBit(SL_DB_SKELETON) || node->drawBit(SL_DB_SKELETON))
                {
                    // Draw axis of the skeleton joints and its parent bones
                    const SLSkeleton* skeleton = node->skeleton();
                    if (skeleton)
                    {   for (auto joint : skeleton->joints())
                        {   
                            // Get the node wm & apply the joints wm
                            SLMat4f wm = node->updateAndGetWM();
                            wm *= joint->updateAndGetWM();

                            // Get parent node wm & apply the parent joint wm
                            SLMat4f parentWM;
                            if (joint->parent())
                            {   parentWM = node->parent()->updateAndGetWM();
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
        }
    }

    GET_GL_ERROR;        // Check if any OGL errors occurred
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw2DGL draws GUI tree in ortho projection. So far no
update is done to the 2D scenegraph.
*/
void SLSceneView::draw2DGL()
{
    SLScene* s = SLScene::current;
    SLfloat startMS = s->timeMilliSec();

    if (!_showMenu &&
        !_showInfo &&
        !_showLoading &&
        !(_showStatsTiming || _showStatsRenderer ||
          _showStatsCamera || _showStatsMemory || _showStatsVideo) &&
        _touchDowns==0 &&
        !_mouseDownL &&
        !_mouseDownM)
    {
        _draw2DTimeMS = s->timeMilliSec() - startMS;
        return;
    }
    
    SLfloat w2 = (SLfloat)_scrWdiv2;
    SLfloat h2 = (SLfloat)_scrHdiv2;
   
    // Set orthographic projection with 0,0,0 in the screen center
    // for now we just have one special GUI case for side by side HMD stereo rendering
    if (_camera && _camera->projection() != P_stereoSideBySideD)
    {        
        // @todo this doesn't need to be done every frame, we can save the current ortho matrix and update on resize
        _stateGL->projectionMatrix.ortho(-w2, w2,-h2, h2, 1.0f, -1.0f);
        // Set viewport over entire screen
        _stateGL->viewport(0, 0, _scrW, _scrH);

        draw2DGLAll();
    }
    else
    {
        // left eye
        _stateGL->projectionMatrix.setMatrix(s->oculus()->orthoProjection(ET_left));
        // Set viewport over entire screen
        _stateGL->viewport(0, 0, _oculusFB.halfWidth(), _oculusFB.height());
        
        draw2DGLAll();

        // left eye
        _stateGL->projectionMatrix.setMatrix(s->oculus()->orthoProjection(ET_right));
        // Set viewport over entire screen
        _stateGL->viewport(_oculusFB.halfWidth(), 0, _oculusFB.halfWidth(), _oculusFB.height());
        
        draw2DGLAll();
        
        // temp visualization of the texture above
        /*
        glClear(GL_COLOR_BUFFER_BIT);
        static SLGLGenericProgram tmpShader("StereoOculus.vert", "StereoOculus.frag");

        static GLuint screenQuad = 0;
        if (!screenQuad) 
        {   GLfloat quadVerts[] = {-1, -1,
                                    1, -1,
                                   -1,  1,
                                    1,  1};
            glGenBuffers(1, &screenQuad);
            glBindBuffer(GL_ARRAY_BUFFER, screenQuad);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
                                 
        glDisable(GL_DEPTH_TEST);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _oculusFB.texID());

        tmpShader.beginUse(); //bind the rift shader

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, screenQuad);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray(0);
        glEnable(GL_DEPTH_TEST);
        */
    }
    
    // below is the normal menu to test interaction with the default mouse
    /*
    _stateGL->projectionMatrix.ortho(-w2, w2,-h2, h2, 1.0f, -1.0f);
    _stateGL->viewport(0, 0, _scrW, _scrH);

   
    _stateGL->depthMask(false);         // Freeze depth buffer for blending
    _stateGL->depthTest(false);         // Disable depth testing
    _stateGL->blend(true);              // Enable blending
    _stateGL->polygonLine(false);       // Only filled polygons

    // Draw menu buttons tree
    if (!_showLoading && _showMenu && s->menu2D())
    {  _stateGL->modelViewMatrix.identity();
        _stateGL->modelViewMatrix.translate(-w2, -h2, 0);
        s->menu2D()->drawRec(this);
    }   
    _stateGL->blend(false);       // turn off blending
    _stateGL->depthMask(true);    // enable depth buffer writing
    _stateGL->depthTest(true);    // enable depth testing
    GET_GL_ERROR;                 // check if any OGL errors occured
    */
    

   _draw2DTimeMS = s->timeMilliSec() - startMS;
   return;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw2DGLAll draws GUI tree in ortho projection.
*/
void SLSceneView::draw2DGLAll()
{
    SLScene* s = SLScene::current;
    SLfloat w2 = (SLfloat)_scrWdiv2;
    SLfloat h2 = (SLfloat)_scrHdiv2;
    SLfloat depth = 1.0f;               // Render depth between -1 & 1

    _stateGL->pushModelViewMatrix();
    _stateGL->modelViewMatrix.identity();

    if (_camera->projection() == P_stereoSideBySideD)
    {
        //@todo the line below will squish the menu. It is the best solution we have currently
        //       that requires no changes to other portions of the menu building. 
        //       but it should be addressed in the future.
        _stateGL->modelViewMatrix.scale(0.5f, 1, 1);
        //w2 *= 0.5f; // we could also just half the with again and comment out the squish above

        // for the time being we simply upscale the UI by the factor of the HMD
        _stateGL->modelViewMatrix.scale(s->oculus()->resolutionScale());
    }

    _stateGL->depthMask(false);         // Freeze depth buffer for blending
    _stateGL->depthTest(false);         // Disable depth testing
    _stateGL->blend(true);              // Enable blending
    _stateGL->polygonLine(false);       // Only filled polygons

    // Draw 2D loading text
    if (_showLoading)
    {   build2DInfoLoading();
        if (s->infoLoading())
        {   _stateGL->pushModelViewMatrix();            
            _stateGL->modelViewMatrix.translate(-w2, h2, depth);
            _stateGL->modelViewMatrix.multiply(s->infoLoading()->om());
            s->infoLoading()->drawRec(this);
            _stateGL->popModelViewMatrix();
        }
    }

    // Draw statistics for GL
    if (!_showLoading &&
        (_showStatsTiming || _showStatsRenderer || _showStatsCamera || _showStatsMemory || _showStatsVideo) &&
        (s->menu2D()==s->menuGL() || s->menu2D()==s->btnAbout()))
    {   build2DInfoGL();
        if (s->infoGL())
		{   _stateGL->pushModelViewMatrix();
			_stateGL->modelViewMatrix.translate(-w2, h2, depth);
			_stateGL->modelViewMatrix.translate(SLButton::minMenuPos.x, -SLButton::minMenuPos.y, 0);
            _stateGL->modelViewMatrix.multiply(s->infoGL()->om());
            s->infoGL()->drawRec(this);
            _stateGL->popModelViewMatrix();
        }
    }
   
    // Draw statistics for RT
    if (!_showLoading &&
       (_showStatsTiming || _showStatsRenderer || _showStatsCamera || _showStatsMemory || _showStatsVideo) &&
        (s->menu2D()==s->menuRT()))
    {   build2DInfoRT();
        if (s->infoRT()) 
        {   _stateGL->pushModelViewMatrix();  
            _stateGL->modelViewMatrix.translate(-w2, h2, depth);
            _stateGL->modelViewMatrix.multiply(s->infoRT()->om());
            s->infoRT()->drawRec(this);
            _stateGL->popModelViewMatrix();
        } 
    }

    // Draw scene info text if menuGL or menuRT is closed
    if (!_showLoading && _showInfo && 
        s->info() && !s->info()->text().empty() &&
        _camera->projection()<=P_monoOrthographic &&
        (s->menu2D()==s->menuGL() || 
         s->menu2D()==s->menuRT() ||
         s->menu2D()==s->menuPT()) && SLButton::buttonParent==nullptr)
    {
        _stateGL->pushModelViewMatrix();  
        _stateGL->modelViewMatrix.translate(-w2, -h2, depth);
        _stateGL->modelViewMatrix.multiply(s->info()->om());
        s->info()->drawRec(this);
        _stateGL->popModelViewMatrix();
    }
   
    // Draw menu buttons tree
    if (!_showLoading && _showMenu && s->menu2D())
    {   _stateGL->pushModelViewMatrix();  
        _stateGL->modelViewMatrix.translate(-w2, -h2, 0);
        s->menu2D()->drawRec(this);
        _stateGL->popModelViewMatrix();
    }   
   
    // 2D finger touch points  
    #ifndef SL_GLES2
    if (_touchDowns)
    {   _stateGL->multiSample(true);
        _stateGL->pushModelViewMatrix();  
      
        // Go to lower-left screen corner
        _stateGL->modelViewMatrix.translate(-w2, -h2, depth);
      
        SLVVec3f touch;
        touch.resize(_touchDowns);
        for (SLint i=0; i<_touchDowns; ++i)
        {   touch[i].x = (SLfloat)_touch[i].x;
            touch[i].y = (SLfloat)(_scrH - _touch[i].y);
            touch[i].z = 0.0f;
        }
      
        _vaoTouch.generateVertexPos(&touch);
      
        SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);
        _vaoTouch.drawArrayAsColored(PT_points, yelloAlpha, 21);
        _stateGL->popModelViewMatrix();
    }
    #endif

    // Draw turntable rotation point
    if ((_mouseDownL || _mouseDownM) && _touchDowns==0)
    {   if (_camera->camAnim()==CA_turntableYUp || _camera->camAnim()==CA_turntableZUp)
        {   _stateGL->multiSample(true);
            _stateGL->pushModelViewMatrix();  
            _stateGL->modelViewMatrix.translate(0, 0, depth);
            SLVVec3f cross = {{0,0,0}};
            _vaoTouch.generateVertexPos(&cross);
            SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);
            _vaoTouch.drawArrayAsColored(PT_points, yelloAlpha, (SLfloat)SL::dpi/12.0f);
            _stateGL->popModelViewMatrix();
        }
    }

    // Draw virtual mouse cursor if we're in HMD stereo mode
    if (_camera->projection() == P_stereoSideBySideD)
    {
        SLfloat hCur = (SLfloat)s->texCursor()->height();
        _stateGL->multiSample(true);
        _stateGL->pushModelViewMatrix();  
        _stateGL->modelViewMatrix.translate(-w2, -h2, 0);
        _stateGL->modelViewMatrix.translate((SLfloat)_posCursor.x, 
                                            (_scrH-_posCursor.y-hCur), 0);
        s->texCursor()->drawSprite();
        _stateGL->popModelViewMatrix();
    }

    _stateGL->popModelViewMatrix();        

    _stateGL->blend(false);       // turn off blending
    _stateGL->depthMask(true);    // enable depth buffer writing
    _stateGL->depthTest(true);    // enable depth testing
    GET_GL_ERROR;                 // check if any OGL errors occurred
}
//-----------------------------------------------------------------------------









//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseDown gets called whenever a mouse button gets pressed and
dispatches the event to the currently attached event handler object.
*/
SLbool SLSceneView::onMouseDown(SLMouseButton button, 
                                SLint x, SLint y, SLKey mod)
{
    SLScene* s = SLScene::current;
   
    // Check first if mouse down was on a button    
    if (s->menu2D() && s->menu2D()->onMouseDown(button, x, y, mod))
        return true;

    // if menu is open close it
    if (SLButton::buttonParent && s->menu2D())
        s->menu2D()->closeAll();
   
    _mouseDownL = (button == MB_left);
    _mouseDownR = (button == MB_right);
    _mouseDownM = (button == MB_middle);
    _mouseMod = mod;
   
    SLbool result = false;
    if (_camera && s->root3D())
    {   result = _camera->onMouseDown(button, x, y, mod);
        for (auto eh : s->eventHandlers())
        {   if (eh->onMouseDown(button, x, y, mod))
                result = true;
        }
    } 
    
    // Grab image during calibration if calibration stream is running
    if (s->calibration().state() == CS_calibrateStream)
        s->calibration().state(CS_calibrateGrab);

    return result;
}  
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseUp gets called whenever a mouse button gets released.
*/
SLbool SLSceneView::onMouseUp(SLMouseButton button, 
                              SLint x, SLint y, SLKey mod)
{  
    SLScene* s = SLScene::current;
    _touchDowns = 0;
   
    if (_raytracer.state()==rtMoveGL)
    {   _renderType = RT_rt;
        _raytracer.state(rtReady);
    }   
   
    // Check first if mouse up was on a button    
    if (s->menu2D() && s->menu2D()->onMouseUp(button, x, y, mod))
        return true;
           
    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;

    if (_camera && s->root3D())
    {   SLbool result = false;
        result = _camera->onMouseUp(button, x, y, mod);
        for (auto eh : s->eventHandlers())
        {   if (eh->onMouseUp(button, x, y, mod))
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
SLbool SLSceneView::onMouseMove(SLint x, SLint y)
{     
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    // save cursor position
    _posCursor.set(x, y);

    _touchDowns = 0;
    SLbool result = false;
      
    if (_mouseDownL || _mouseDownR || _mouseDownM)
    {   SLMouseButton btn = _mouseDownL ? MB_left : 
                            _mouseDownR ? MB_right : MB_middle;
      
        // Handle move in RT mode
        if (_renderType == RT_rt && !_raytracer.continuous())
        {   if (_raytracer.state()==rtFinished)
                _raytracer.state(rtMoveGL);
            else
            {   _raytracer.continuous(false);
                s->menu2D(s->menuGL());
                s->menu2D()->hideAndReleaseRec();
                s->menu2D()->drawBits()->off(SL_DB_HIDDEN);
            }
            _renderType = RT_gl;
        }
      
        result = _camera->onMouseMove(btn, x, y, _mouseMod);

        for (auto eh : s->eventHandlers())
        {   if (eh->onMouseMove(btn, x, y, _mouseMod))
                result = true;
        }
    }  
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseWheel gets called whenever the mouse wheel is turned.
The parameter wheelPos is an increasing or decreeing counter number.
*/
SLbool SLSceneView::onMouseWheelPos(SLint wheelPos, SLKey mod)
{  
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    static SLint lastMouseWheelPos = 0;
    SLint delta = wheelPos-lastMouseWheelPos;
    lastMouseWheelPos = wheelPos;
    return onMouseWheel(delta, mod);
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseWheel gets called whenever the mouse wheel is turned.
The parameter delta is positive/negative depending on the wheel direction
*/
SLbool SLSceneView::onMouseWheel(SLint delta, SLKey mod)
{
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    // Handle mouse wheel in RT mode
    if (_renderType == RT_rt && !_raytracer.continuous() && 
        _raytracer.state()==rtFinished)
        _raytracer.state(rtReady);
    SLbool result = false;

    // update active camera
    result = _camera->onMouseWheel(delta, mod);

    for (auto eh : s->eventHandlers())
    {   if (eh->onMouseWheel(delta, mod))
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
                                  SLint x, SLint y, SLKey mod)
{  
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;
   
    // Check first if mouse down was on a button    
    if (s->menu2D() && s->menu2D()->onDoubleClick(button, x, y, mod))
        return true;

    SLbool result = false;
   
    // Do object picking with ray cast
    if (button == MB_left)
    {   _mouseDownR = false;
      
        SLRay pickRay;
        if (_camera) 
        {   _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);
            s->root3D()->hitRec(&pickRay);
            if(pickRay.hitNode)
                cout << "NODE HIT: " << pickRay.hitNode->name() << endl;
        }
      
        if (pickRay.length < FLT_MAX)
        {   s->selectNodeMesh(pickRay.hitNode, pickRay.hitMesh);
            if (onSelectedNodeMesh)
            onSelectedNodeMesh(s->selectedNode(), s->selectedMesh());
            result = true;
        }
      
    } else
    {   result = _camera->onDoubleClick(button, x, y, mod);
        for (auto eh : s->eventHandlers())
        {   if (eh->onDoubleClick(button, x, y, mod))
                result = true;
        }
    }
    return result;
} 
//-----------------------------------------------------------------------------
/*! SLSceneView::onLongTouch gets called when the mouse or touch is down for
more than 500ms and has not moved.
*/
SLbool SLSceneView::onLongTouch(SLint x, SLint y)
{
    //SL_LOG("onLongTouch(%d, %d)\n", x, y);
    return true;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onTouch2Down gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLSceneView::onTouch2Down(SLint x1, SLint y1, SLint x2, SLint y2)
{
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
    _touchDowns = 2;
   
    SLbool result = false;
    result = _camera->onTouch2Down(x1, y1, x2, y2);
    for (auto eh : s->eventHandlers())
    {   if (eh->onTouch2Down(x1, y1, x2, y2))
            result = true;
    }  
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onTouch2Move gets called whenever two fingers touch a handheld
screen.
*/
SLbool SLSceneView::onTouch2Move(SLint x1, SLint y1, SLint x2, SLint y2)
{
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
   
    SLbool result = false;
    if (_touchDowns==2)
    {   result = _camera->onTouch2Move(x1, y1, x2, y2);
        for (auto eh : s->eventHandlers())
        {  if (eh->onTouch2Move(x1, y1, x2, y2))
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
SLbool SLSceneView::onTouch2Up(SLint x1, SLint y1, SLint x2, SLint y2)
{
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
    _touchDowns = 0;
    SLbool result = false;
   
    result = _camera->onTouch2Up(x1, y1, x2, y2);
    for (auto eh : s->eventHandlers())
    {   if (eh->onTouch2Up(x1, y1, x2, y2))
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
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;
    
    if (key == '5') { _camera->unitScaling(_camera->unitScaling()+0.1f); SL_LOG("New unit scaling: %f", _camera->unitScaling()); return true; }
    if (key == '6') { _camera->unitScaling(_camera->unitScaling()-0.1f); SL_LOG("New unit scaling: %f", _camera->unitScaling()); return true; }
    if (key == '7') return onCommand(C_dpiInc);
    if (key == '8') return onCommand(C_dpiDec);

    if (key=='N') return onCommand(C_normalsToggle);
    if (key=='P') return onCommand(C_wireMeshToggle);
    if (key=='C') return onCommand(C_faceCullToggle);
    if (key=='T') return onCommand(C_textureToggle);
    if (key=='M') return onCommand(C_multiSampleToggle);
    if (key=='F') return onCommand(C_frustCullToggle);
    if (key=='B') return onCommand(C_bBoxToggle);

    if (key==K_esc)
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

    SLbool result = false;
    if (key || mod)
    {   result = _camera->onKeyPress(key, mod);
        for (auto eh : s->eventHandlers())
        {   if (eh->onKeyPress(key, mod))
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
    SLScene* s = SLScene::current;
    if (!s->root3D()) return false;

    SLbool result = false;
   
    if (key || mod)
    {   result = _camera->onKeyRelease(key, mod);
        for (auto eh : s->eventHandlers())
        {  if (eh->onKeyRelease(key, mod))
                result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onRotation: Event handler for rotation change of a mobile
device with Euler angles for pitch, yaw and roll. 
With the parameter zeroYawAfterSec sets the time in seconds after
which the yaw angle is set to zero by subtracting the average yaw in this time.
*/
void SLSceneView::onRotationPYR(SLfloat pitchRAD, 
                                SLfloat yawRAD, 
                                SLfloat rollRAD,
                                SLfloat zeroYawAfterSec)
{
    SLScene* s = SLScene::current;
    if (!s->root3D()) return;

    SL_LOG("onRotation: pitch: %3.1f, yaw: %3.1f, roll: %3.1f\n",
           pitchRAD * SL_RAD2DEG,
           yawRAD   * SL_RAD2DEG,
           rollRAD  * SL_RAD2DEG);

    // Set the yaw to zero by subtracting the averaged yaw after the passed NO. of sec.
    // Array of 60 yaw values for averaging
    static SLAvgFloat initialYaw(60);

    if (zeroYawAfterSec == 0.0f)
    {   _deviceRotation.fromEulerAngles(pitchRAD,yawRAD,rollRAD);
    } else 
    if (SLScene::current->timeSec() < zeroYawAfterSec)
    {   initialYaw.set(yawRAD);
        _deviceRotation.fromEulerAngles(pitchRAD,yawRAD,rollRAD);
    } else
    {  _deviceRotation.fromEulerAngles(pitchRAD,yawRAD-initialYaw.average(),rollRAD);
    }
    _camera->rotation(_deviceRotation);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onRotation: Event handler for rotation change of a mobile
device with rotation quaternion.
*/
void SLSceneView::onRotationQUAT(SLfloat quatX, 
                                 SLfloat quatY, 
                                 SLfloat quatZ, 
                                 SLfloat quatW)
{
    SLScene* s = SLScene::current;
    if (!s->root3D()) return;

    _deviceRotation.set(quatX, quatY, quatZ, quatW);
}

//-----------------------------------------------------------------------------
/*!
SLSceneView::onCommand: Event handler for commands. Most key press or menu
commands are collected and dispatched here.
*/
SLbool SLSceneView::onCommand(SLCommand cmd)
{
    SLScene* s = SLScene::current;

    switch (cmd)
    {
        case C_quit:
            slShouldClose(true);
        case C_menu:
            return false;
        case C_aboutToggle:
            if (s->menu2D())
            {   if (s->menu2D() == s->menuGL())
                    s->menu2D(s->btnAbout());
                else s->menu2D(s->menuGL());
                return true;
            }
            else return false;
        case C_helpToggle:
            if (s->menu2D())
            {   if (s->menu2D() == s->menuGL())
                    s->menu2D(s->btnHelp());
                else s->menu2D(s->menuGL());
                return true;
            }
            else return false;
        case C_creditsToggle:
            if (s->menu2D())
            {   if (s->menu2D() == s->menuGL())
                    s->menu2D(s->btnCredits());
                else s->menu2D(s->menuGL());
                return true;
            }
            else return false;
        case C_noCalibToggle:
            if (s->menu2D())
            {   if (SL::currentSceneID != C_sceneTrackChessboard)
                    s->onLoad(this, (SLCommand)C_sceneEmpty); 
                s->menu2D(s->menuGL());
                return true;
            }
            else return false;
        case C_dpiInc:
            if (SL::dpi < 500)
            {   SL::dpi = (SLint)((SLfloat)SL::dpi * 1.1f);
                rebuild2DMenus(false);
                return true;
            } else return false;
        case C_dpiDec:
            if (SL::dpi > 140)
            {   SL::dpi = (SLint)((SLfloat)SL::dpi * 0.9f);
                rebuild2DMenus(false);
                return true;
            } else return false;

        case C_sceneMinimal:
        case C_sceneFigure:
        case C_sceneLargeModel:
        case C_sceneMeshLoad:
        case C_sceneVRSizeTest:
        case C_sceneChristoffel:
        case C_sceneRevolver:
        case C_sceneTextureBlend:
        case C_sceneTextureFilter:
        case C_sceneTextureVideo:
        case C_sceneFrustumCull:
        case C_sceneMassiveData:

        case C_scenePerVertexBlinn:
        case C_scenePerPixelBlinn:
        case C_scenePerVertexWave:
        case C_sceneWater:
        case C_sceneBumpNormal:
        case C_sceneBumpParallax:
        case C_sceneEarth:

        case C_sceneMassAnimation:
        case C_sceneNodeAnimation:
        case C_sceneSkeletalAnimation:
        case C_sceneAstroboyArmy:

        case C_sceneTrackChessboard:
        case C_sceneTrackAruco:
        case C_sceneTrackFeatures2D:

        case C_sceneRTSpheres:
        case C_sceneRTMuttenzerBox:
        case C_sceneRTSoftShadows:
        case C_sceneRTDoF:
        case C_sceneRTTest:
        case C_sceneRTLens:        s->onLoad(this, (SLCommand)cmd); return false;

        case C_useSceneViewCamera: switchToSceneViewCamera(); return true;

        case C_statsTimingToggle:  _showStatsTiming = !_showStatsTiming; return true;
        case C_statsRendererToggle:_showStatsRenderer = !_showStatsRenderer; return true;
        case C_statsMemoryToggle:  _showStatsMemory = !_showStatsMemory; return true;
        case C_statsCameraToggle:  _showStatsCamera = !_showStatsCamera; return true;
        case C_statsVideoToggle:   _showStatsVideo = !_showStatsVideo; return true;

        case C_sceneInfoToggle:    _showInfo = !_showInfo; return true;
        case C_waitEventsToggle:   _waitEvents = !_waitEvents; return true;
        case C_multiSampleToggle:
            _doMultiSampling = !_doMultiSampling;
            _raytracer.aaSamples(_doMultiSampling ? 3 : 1);
            return true;
        case C_frustCullToggle:    _doFrustumCulling = !_doFrustumCulling; return true;
        case C_depthTestToggle:    _doDepthTest = !_doDepthTest; return true;

        case C_normalsToggle:      _drawBits.toggle(SL_DB_NORMALS);  return true;
        case C_wireMeshToggle:     _drawBits.toggle(SL_DB_WIREMESH); return true;
        case C_bBoxToggle:         _drawBits.toggle(SL_DB_BBOX);     return true;
        case C_axisToggle:         _drawBits.toggle(SL_DB_AXIS);     return true;
        case C_skeletonToggle:     _drawBits.toggle(SL_DB_SKELETON); return true;
        case C_voxelsToggle:       _drawBits.toggle(SL_DB_VOXELS);   return true;
        case C_faceCullToggle:     _drawBits.toggle(SL_DB_CULLOFF);  return true;
        case C_textureToggle:      _drawBits.toggle(SL_DB_TEXOFF);   return true;

        case C_animationToggle:     s->stopAnimations(!s->stopAnimations()); return true;
        case C_clearCalibration:    s->calibration().state(CS_uncalibrated); 
                                    s->onLoad(this, C_sceneTrackChessboard); return false;
        case C_renderOpenGL:
            _renderType = RT_gl;
            s->menu2D(s->menuGL());
            return true;
        case C_rtContinuously:
            _raytracer.continuous(!_raytracer.continuous());
            return true;
        case C_rtDistributed:
            _raytracer.distributed(!_raytracer.distributed());
            startRaytracing(5);
            return true;
        case C_rt1: startRaytracing(1); return true;
        case C_rt2: startRaytracing(2); return true;
        case C_rt3: startRaytracing(3); return true;
        case C_rt4: startRaytracing(4); return true;
        case C_rt5: startRaytracing(5); return true;
        case C_rt6: startRaytracing(6); return true;
        case C_rt7: startRaytracing(7); return true;
        case C_rt8: startRaytracing(8); return true;
        case C_rt9: startRaytracing(9); return true;
        case C_rt0: startRaytracing(0); return true;
        case C_rtSaveImage: _raytracer.saveImage(); return true;

        case C_pt1: startPathtracing(5, 1); return true;
        case C_pt10: startPathtracing(5, 10); return true;
        case C_pt50: startPathtracing(5, 50); return true;
        case C_pt100: startPathtracing(5, 100); return true;
        case C_pt500: startPathtracing(5, 500); return true;
        case C_pt1000: startPathtracing(5, 1000); return true;
        case C_pt5000: startPathtracing(5, 5000); return true;
        case C_pt10000: startPathtracing(5, 100000); return true;
        case C_ptSaveImage: _pathtracer.saveImage(); return true;

        default: break;
    }

    if (_camera)
    {
        SLProjection prevProjection = _camera->projection();
        SLbool perspectiveChanged = prevProjection != (SLProjection)(cmd - C_projPersp);

        switch (cmd)
        {
            case C_projPersp:
                _camera->projection(P_monoPerspective);
                if (_renderType == RT_rt && !_raytracer.continuous() &&
                    _raytracer.state() == rtFinished)
                    _raytracer.state(rtReady);
                break;
            case C_projOrtho:
                _camera->projection(P_monoOrthographic);
                if (_renderType == RT_rt && !_raytracer.continuous() &&
                    _raytracer.state() == rtFinished)
                    _raytracer.state(rtReady);
                break;
            case C_projSideBySide:    _camera->projection(P_stereoSideBySide); break;
            case C_projSideBySideP:   _camera->projection(P_stereoSideBySideP); break;
            case C_projSideBySideD:   _camera->projection(P_stereoSideBySideD); break;
            case C_projLineByLine:    _camera->projection(P_stereoLineByLine); break;
            case C_projColumnByColumn:_camera->projection(P_stereoColumnByColumn); break;
            case C_projPixelByPixel:  _camera->projection(P_stereoPixelByPixel); break;
            case C_projColorRC:       _camera->projection(P_stereoColorRC); break;
            case C_projColorRG:       _camera->projection(P_stereoColorRG); break;
            case C_projColorRB:       _camera->projection(P_stereoColorRB); break;
            case C_projColorYB:       _camera->projection(P_stereoColorYB); break;

            case C_camSpeedLimitInc:  _camera->maxSpeed(_camera->maxSpeed()*1.2f); return true;
            case C_camSpeedLimitDec:  _camera->maxSpeed(_camera->maxSpeed()*0.8f); return true;
            case C_camEyeSepInc:      _camera->onMouseWheel(1, K_ctrl); return true;
            case C_camEyeSepDec:      _camera->onMouseWheel(-1, K_ctrl); return true;
            case C_camFocalDistInc:   _camera->onMouseWheel(1, K_shift); return true;
            case C_camFocalDistDec:   _camera->onMouseWheel(-1, K_shift); return true;
            case C_camFOVInc:         _camera->onMouseWheel(1, K_alt); return true;
            case C_camFOVDec:         _camera->onMouseWheel(-1, K_alt); return true;
            case C_camAnimTurnYUp:    _camera->camAnim(CA_turntableYUp); return true;
            case C_camAnimTurnZUp:    _camera->camAnim(CA_turntableZUp); return true;
            case C_camAnimWalkYUp:    _camera->camAnim(CA_walkingYUp); return true;
            case C_camAnimWalkZUp:    _camera->camAnim(CA_walkingZUp); return true;
            case C_camDeviceRotOn:    _camera->useDeviceRot(true); return true;
            case C_camDeviceRotOff:   _camera->useDeviceRot(false); return true;
            case C_camDeviceRotToggle:_camera->useDeviceRot(!_camera->useDeviceRot()); return true;
            case C_camReset:          _camera->resetToInitialState(); return true;
            default: break;
        }

        // special treatment for the menu position in side-by-side projection
        if (perspectiveChanged)
        {   if (cmd == C_projSideBySideD)
            {   _vrMode = true;
                SL::dpi *= 2;
                SLButton::minMenuPos.set(_scrW*0.25f + 100.0f, _scrH*0.5f - 150.0f);
                rebuild2DMenus();
                if (onShowSysCursor)
                    onShowSysCursor(false);
            }
            else if (prevProjection == P_stereoSideBySideD)
            {   _vrMode = false;
                SL::dpi /= 2;               
                SLButton::minMenuPos.set(10.0f, 10.0f);
                rebuild2DMenus();
                if (onShowSysCursor)
                    onShowSysCursor(true);
            }
        }
    }

    return false;
}
//-----------------------------------------------------------------------------






//-----------------------------------------------------------------------------
/*! 
SLSceneView::rebuild2DMenus force a rebuild of all 2d elements, might be needed
if dpi or other screenspace related parameters changed.
@todo the menu is still contained in the scene which partly breaks this behavior
      for multiview applications.
*/
void SLSceneView::rebuild2DMenus(SLbool showAboutFirst)
{
    SLScene* s = SLScene::current;
    SLstring infoText = s->info()->text();

    s->deleteAllMenus();
    build2DMenus();
    if (!showAboutFirst)
        s->menu2D(s->menuGL());

    if (_showInfo)
        s->info(this, infoText);
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::build2D builds the GUI menu button tree for the _menuGL and the 
_menuRT group as well as the _infoGL and _infoRT groups with info texts. 
See SLButton and SLText class for more infos. 
*/
void SLSceneView::build2DMenus()
{  
    SLScene* s = SLScene::current;
   
    // Create menus only once
    if (s->menu2D()) return;

    // Get current camera projection
    SLProjection proj = _camera ? _camera->projection() : P_monoPerspective;
   
    // Get current camera animation
    SLCamAnim anim = _camera ? _camera->camAnim() : CA_turntableYUp;

    // Get current camera device rotation usage
    SLbool useDeviceRot = _camera ? _camera->useDeviceRot() : true;

    // Set font size depending on DPI
    SLTexFont* f = SLTexFont::getFont(1.7f, SL::dpi);

    SLButton *mn1, *mn2, *mn3, *mn4, *mn5;  // sub menu button pointer
    SLCommand curS = SL::currentSceneID;   // current scene number
   
    mn1 = new SLButton(this, ">", f, C_menu, false, false, 0, true, 0, 0, SLCol3f::COLBFH, 0.3f, TA_centerCenter); mn1->drawBits()->off(SL_DB_HIDDEN);

        mn2 = new SLButton(this, "Load Scene >", f); mn1->addChild(mn2);
   
            mn3 = new SLButton(this, "General >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Minimal Scene", f, C_sceneMinimal, true, curS==C_sceneMinimal, mn2));
            SLstring large1 = SLImporter::defaultPath + "PLY/xyzrgb_dragon.ply";
            SLstring large2 = SLImporter::defaultPath + "PLY/mesh_zermatt.ply";
            SLstring large3 = SLImporter::defaultPath + "PLY/switzerland.ply";
            if(SLFileSystem::fileExists(large1) ||
               SLFileSystem::fileExists(large2) ||
               SLFileSystem::fileExists(large3))
                mn3->addChild(new SLButton(this, "Large Model", f, C_sceneLargeModel, true, curS==C_sceneLargeModel, mn2));
            mn3->addChild(new SLButton(this, "Figure", f, C_sceneFigure, true, curS==C_sceneFigure, mn2));
            mn3->addChild(new SLButton(this, "Mesh Loader", f, C_sceneMeshLoad, true, curS == C_sceneMeshLoad, mn2));
            mn3->addChild(new SLButton(this, "Texture Blending", f, C_sceneTextureBlend, true, curS==C_sceneTextureBlend, mn2));
            mn3->addChild(new SLButton(this, "Texture Filters and 3D texture", f, C_sceneTextureFilter, true, curS==C_sceneTextureFilter, mn2));
            mn3->addChild(new SLButton(this, "Frustum Culling", f, C_sceneFrustumCull, true, curS==C_sceneFrustumCull, mn2));
            mn3->addChild(new SLButton(this, "Massive Data Scene", f, C_sceneMassiveData, true, curS==C_sceneMassiveData, mn2));

            mn3 = new SLButton(this, "Shader >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Per Vertex Lighting", f, C_scenePerVertexBlinn, true, curS==C_scenePerVertexBlinn, mn2));
            mn3->addChild(new SLButton(this, "Per Pixel Lighting", f, C_scenePerPixelBlinn, true, curS==C_scenePerPixelBlinn, mn2));
            mn3->addChild(new SLButton(this, "Per Vertex Wave", f, C_scenePerVertexWave, true, curS==C_scenePerVertexWave, mn2));
            mn3->addChild(new SLButton(this, "Water", f, C_sceneWater, true, curS==C_sceneWater, mn2));
            mn3->addChild(new SLButton(this, "Bump Mapping", f, C_sceneBumpNormal, true, curS==C_sceneBumpNormal, mn2, true));
            mn3->addChild(new SLButton(this, "Parallax Mapping", f, C_sceneBumpParallax, true, curS==C_sceneBumpParallax, mn2));
            mn3->addChild(new SLButton(this, "Glass Shader", f, C_sceneRevolver, true, curS==C_sceneRevolver, mn2));
            mn3->addChild(new SLButton(this, "Earth Shader", f, C_sceneEarth, true, curS==C_sceneEarth, mn2));

            mn3 = new SLButton(this, "Animation >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Mass Animation", f, C_sceneMassAnimation, true, curS==C_sceneMassAnimation, mn2));
            mn3->addChild(new SLButton(this, "Astroboy Army", f, C_sceneAstroboyArmy, true, curS==C_sceneAstroboyArmy, mn2));
            mn3->addChild(new SLButton(this, "Skeletal Animation", f, C_sceneSkeletalAnimation, true, curS==C_sceneSkeletalAnimation, mn2));
            mn3->addChild(new SLButton(this, "Node Animation", f, C_sceneNodeAnimation, true, curS==C_sceneNodeAnimation, mn2));
    
            mn3 = new SLButton(this, "Using Video >", f); mn2->addChild(mn3);
            //mn3->addChild(new SLButton(this, "Track or Create 2D-Feature Marker", f, C_sceneTrackFeatures2D, true, curS==C_sceneTrackFeatures2D, mn2));
            mn3->addChild(new SLButton(this, "Track ArUco Marker", f, C_sceneTrackAruco, true, curS==C_sceneTrackAruco, mn2));
            mn3->addChild(new SLButton(this, "Track Chessboard or Calibrate Camera", f, C_sceneTrackChessboard, true, curS==C_sceneTrackChessboard, mn2));
            mn3->addChild(new SLButton(this, "Clear Camera Calibration", f, C_clearCalibration, false, false, mn2));
            mn3->addChild(new SLButton(this, "Christoffel Tower", f, C_sceneChristoffel, true, curS == C_sceneChristoffel, mn2));
            mn3->addChild(new SLButton(this, "Texture from live video", f, C_sceneTextureVideo, true, curS==C_sceneTextureVideo, mn2));
   
            mn3 = new SLButton(this, "Ray tracing >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Spheres", f, C_sceneRTSpheres, true, curS==C_sceneRTSpheres, mn2));
            mn3->addChild(new SLButton(this, "Muttenzer Box", f, C_sceneRTMuttenzerBox, true, curS==C_sceneRTMuttenzerBox, mn2));
            mn3->addChild(new SLButton(this, "Soft Shadows", f, C_sceneRTSoftShadows, true, curS==C_sceneRTSoftShadows, mn2));
            mn3->addChild(new SLButton(this, "Depth of Field", f, C_sceneRTDoF, true, curS==C_sceneRTDoF, mn2));
            mn3->addChild(new SLButton(this, "Lens Test", f, C_sceneRTLens, true, curS == C_sceneRTLens, mn2));
            mn3->addChild(new SLButton(this, "RT Test", f, C_sceneRTTest, true, curS == C_sceneRTTest, mn2));

            mn3 = new SLButton(this, "Path tracing >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Muttenzer Box", f, C_sceneRTMuttenzerBox, true, curS==C_sceneRTMuttenzerBox, mn2));

        mn2 = new SLButton(this, "Camera >", f); mn1->addChild(mn2);
        mn2->addChild(new SLButton(this, "Reset", f, C_camReset));

            stringstream ss;  ss << "UI-Resolution (DPI: " << SL::dpi << ") >";
            mn3 = new SLButton(this, ss.str(), f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "+10%", f, C_dpiInc));
            mn3->addChild(new SLButton(this, "-10%", f, C_dpiDec));
    
            mn3 = new SLButton(this, "Projection >", f); mn2->addChild(mn3);
            for (SLint p=P_monoPerspective; p<=P_monoOrthographic; ++p)
            {   mn3->addChild(new SLButton(this, SLCamera::projectionToStr((SLProjection)p), f,
                                           (SLCommand)(C_projPersp+p), true, proj==p, mn3));
            }
                mn4 = new SLButton(this, "Stereo >", f); mn3->addChild(mn4);

                    mn5 = new SLButton(this, "Eye separation >", f); mn4->addChild(mn5);
                    mn5->addChild(new SLButton(this, "-10%", f, C_camEyeSepDec, false, false, 0, false));
                    mn5->addChild(new SLButton(this, "+10%", f, C_camEyeSepInc, false, false, 0, false));

                    mn5 = new SLButton(this, "Focal dist. >", f); mn4->addChild(mn5);
                    mn5->addChild(new SLButton(this, "+5%", f, C_camFocalDistInc, false, false, 0, false));
                    mn5->addChild(new SLButton(this, "-5%", f, C_camFocalDistDec, false, false, 0, false));

                for (SLint p=P_stereoSideBySide; p<=P_stereoColorYB; ++p)
                {   mn4->addChild(new SLButton(this, SLCamera::projectionToStr((SLProjection)p), f,
                                               (SLCommand)(C_projPersp+p), true, proj==p, mn3));
                }

            mn3 = new SLButton(this, "Animation >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Walking Y up",   f, C_camAnimWalkYUp, true, anim==CA_walkingYUp, mn3));
            mn3->addChild(new SLButton(this, "Walking Z up",   f, C_camAnimWalkZUp, true, anim==CA_walkingYUp, mn3));
            mn3->addChild(new SLButton(this, "Turntable Y up", f, C_camAnimTurnYUp, true, anim==CA_turntableYUp, mn3));
            mn3->addChild(new SLButton(this, "Turntable Z up", f, C_camAnimTurnZUp, true, anim==CA_turntableZUp, mn3));

            mn3 = new SLButton(this, "View Angle >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "+5 deg.", f, C_camFOVInc, false, false, 0, false));
            mn3->addChild(new SLButton(this, "-5 deg.", f, C_camFOVDec, false, false, 0, false));

            mn3 = new SLButton(this, "Walk Speed >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "+20%", f, C_camSpeedLimitInc, false, false, 0, false));
            mn3->addChild(new SLButton(this, "-20%", f, C_camSpeedLimitDec, false, false, 0, false));

        mn2->addChild(new SLButton(this, "Use Device Rotation", f, C_camDeviceRotToggle, true, useDeviceRot, 0, false));

        mn2 = new SLButton(this, "Render States >", f); mn1->addChild(mn2);
        mn2->addChild(new SLButton(this, "Slowdown on Idle", f, C_waitEventsToggle, true, _waitEvents, 0, false));
        if (_stateGL->hasMultiSampling())
            mn2->addChild(new SLButton(this, "Do Multi Sampling", f, C_multiSampleToggle, true, _doMultiSampling, 0, false));
        mn2->addChild(new SLButton(this, "Do Frustum Culling", f, C_frustCullToggle, true, _doFrustumCulling, 0, false));
        mn2->addChild(new SLButton(this, "Do Depth Test", f, C_depthTestToggle, true, _doDepthTest, 0, false));
        mn2->addChild(new SLButton(this, "Animation off", f, C_animationToggle, true, false, 0, false));

        mn2 = new SLButton(this, "Render Flags >", f); mn1->addChild(mn2);
        mn2->addChild(new SLButton(this, "Textures off", f, C_textureToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "Back Faces", f, C_faceCullToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "Skeleton", f, C_skeletonToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "AABB", f, C_bBoxToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "Axis", f, C_axisToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "Voxels", f, C_voxelsToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "Normals", f, C_normalsToggle, true, false, 0, false));
        mn2->addChild(new SLButton(this, "Wired mesh", f, C_wireMeshToggle, true, false, 0, false));
   
        mn2 = new SLButton(this, "Infos >", f); mn1->addChild(mn2);
        mn2->addChild(new SLButton(this, "About", f, C_aboutToggle));
        mn2->addChild(new SLButton(this, "Help", f, C_helpToggle));

            mn3 = new SLButton(this, "Statistics >", f); mn2->addChild(mn3);
            mn3->addChild(new SLButton(this, "Video",  f, C_statsVideoToggle,  true, _showStatsVideo, 0, false));
            mn3->addChild(new SLButton(this, "Memory", f, C_statsMemoryToggle, true, _showStatsMemory, 0, false));
            mn3->addChild(new SLButton(this, "Camera", f, C_statsCameraToggle, true, _showStatsCamera, 0, false));
            mn3->addChild(new SLButton(this, "Renderer", f, C_statsRendererToggle, true, _showStatsRenderer, 0, false));
            mn3->addChild(new SLButton(this, "Timing", f, C_statsTimingToggle, true, _showStatsTiming, 0, false));

        mn2->addChild(new SLButton(this, "Credits", f, C_creditsToggle));
        mn2->addChild(new SLButton(this, "Scene Info", f, C_sceneInfoToggle, true, _showInfo));

        mn2 = new SLButton(this, "Renderer >", f); mn1->addChild(mn2);
        mn2->addChild(new SLButton(this, "Ray tracing", f, C_rt5, false, false, 0, true));
        #ifndef SL_GLES
        mn2->addChild(new SLButton(this, "Path tracing", f, C_pt10, false, false, 0, true));
        #else
        mn2->addChild(new SLButton(this, "Path tracing", f, C_pt1, false, false, 0, true));
        #endif

        mn2 = new SLButton(this, "Quit", f, C_quit); mn1->addChild(mn2);

    // Init OpenGL menu
    _stateGL->modelViewMatrix.identity();
    mn1->setSizeRec();
    mn1->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    mn1->updateAABBRec();
    s->menuGL(mn1);
   
    // Build ray tracing menu
    SLCol3f green(0.0f,0.5f,0.0f);
   
    mn1 = new SLButton(this, ">", f, C_menu, false, false, 0, true,  0, 0, green, 0.3f, TA_centerCenter);
    mn1->drawBits()->off(SL_DB_HIDDEN);
   
    mn1->addChild(new SLButton(this, "OpenGL Rendering", f, C_renderOpenGL, false, false, 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Render continuously", f, C_rtContinuously, true, _raytracer.continuous(), 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Render parallel distributed", f, C_rtDistributed, true, _raytracer.distributed(), 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Rendering Depth 1", f, C_rt1, false, false, 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Rendering Depth 5", f, C_rt5, false, false, 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Rendering Depth max.", f, C_rt0, false, false, 0, true,  0, 0, green));
    #ifndef SL_GLES2
    mn1->addChild(new SLButton(this, "Save Image", f, C_rtSaveImage, false, false, 0, true,  0, 0, green));
    #endif

    // Init RT menu
    _stateGL->modelViewMatrix.identity();
    mn1->setSizeRec();
    mn1->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    mn1->updateAABBRec();
    s->menuRT(mn1);

    // Build path tracing menu
    SLCol3f blue(0.0f,0.0f,0.5f);
   
    mn1 = new SLButton(this, ">", f, C_menu, false, false, 0, true,  0, 0, blue, 0.3f, TA_centerCenter);
    mn1->drawBits()->off(SL_DB_HIDDEN);

    mn1->addChild(new SLButton(this, "OpenGL Rendering", f, C_renderOpenGL, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "1 Sample Ray", f, C_pt1, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "10 Sample Rays", f, C_pt10, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "50 Sample Rays", f, C_pt50, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "100 Sample Rays", f, C_pt100, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "500 Sample Rays", f, C_pt500, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "1000 Sample Rays", f, C_pt1000, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "5000 Sample Rays", f, C_pt5000, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "10000 Sample Rays", f, C_pt10000, false, false, 0, true,  0, 0, blue));
    #ifndef SL_GLES2
    mn1->addChild(new SLButton(this, "Save Image", f, C_ptSaveImage, false, false, 0, true,  0, 0, blue));
    #endif

    // Init PT menu
    _stateGL->modelViewMatrix.identity();
    mn1->setSizeRec();
    mn1->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    mn1->updateAABBRec();
    s->menuPT(mn1);

    build2DMsgBoxes(); 

    // if menu is initially visible show first the about button
    if (_showMenu)
         s->menu2D(s->btnAbout());
    else s->menu2D(s->menuGL());
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::build2DInfoGL builds the _infoGL groups with info texts. 
See SLButton and SLText class for more infos. 
*/
void SLSceneView::build2DInfoGL()
{
    SLScene* s = SLScene::current;
    if (s->infoGL()) delete s->infoGL();
   
    // prepare some statistic infos
    SLCamera* cam = camera();

    SLchar m[2550];   // message character array
    m[0]=0;           // set zero length

    if (_showStatsTiming)
    {
        SLfloat updateTimePC    = s->updateTimesMS().average()    / s->frameTimesMS().average()*100.0f;
        SLfloat trackingTimePC  = s->trackingTimesMS().average()  / s->frameTimesMS().average()*100.0f;
        SLfloat cullTimePC      = s->cullTimesMS().average()      / s->frameTimesMS().average()*100.0f;
        SLfloat draw3DTimePC    = s->draw3DTimesMS().average()    / s->frameTimesMS().average()*100.0f;
        SLfloat draw2DTimePC    = s->draw2DTimesMS().average()    / s->frameTimesMS().average()*100.0f;
        SLfloat captureTimePC   = s->captureTimesMS().average()   / s->frameTimesMS().average()*100.0f;

        sprintf(m+strlen(m), "Timing -------------------------------------\\n");
        sprintf(m+strlen(m), "Scene: %s\\n", s->name().c_str());
        sprintf(m+strlen(m), "FPS: %4.1f  (Size: %d x %d, DPI: %d)\\n", s->fps(), _scrW, _scrH, SL::dpi);
        sprintf(m+strlen(m), "Frame Time : %4.1f ms (100%%)\\n", s->frameTimesMS().average());
        sprintf(m+strlen(m), "Update Time : %4.1f ms (%0.0f%%)\\n", s->updateTimesMS().average(), updateTimePC);
        sprintf(m+strlen(m), "> Tracking Time: %4.1f ms (%0.0f%%)\\n", s->trackingTimesMS().average(), trackingTimePC);
        sprintf(m+strlen(m), "Culling Time : %4.1f ms (%0.0f%%)\\n", s->cullTimesMS().average(), cullTimePC);
        sprintf(m+strlen(m), "Draw Time 3D: %4.1f ms (%0.0f%%)\\n", s->draw3DTimesMS().average(), draw3DTimePC);
        sprintf(m+strlen(m), "Draw Time 2D: %4.1f ms (%0.0f%%)\\n", s->draw2DTimesMS().average(), draw2DTimePC);
        #ifdef SL_GLSL
        sprintf(m+strlen(m), "Capture Time: %4.1f ms\\n", s->captureTimesMS().average());
        #else
        sprintf(m+strlen(m), "Capture Time: %4.1f ms (%0.0f%%)\\n", s->captureTimesMS().average(), captureTimePC);
        #endif
        sprintf(m+strlen(m), "NO. of drawcalls: %d\\n", SLGLVertexArray::totalDrawCalls);
    }

    if (_showStatsRenderer)
    {
        sprintf(m+strlen(m), "Renderer -----------------------------------\\n");
        sprintf(m+strlen(m), "OpenGL: %s\\n", _stateGL->glVersionNO().c_str());
        sprintf(m+strlen(m), "Vendor: %s\\n", _stateGL->glVendor().c_str());
        sprintf(m+strlen(m), "Renderer: %s\\n", _stateGL->glRenderer().c_str());
        sprintf(m+strlen(m), "GLSL: %s\\n", _stateGL->glSLVersionNO().c_str());
    }

    if (_showStatsCamera)
    {
        sprintf(m+strlen(m), "Camera -------------------------------------\\n");
        sprintf(m+strlen(m), "Projection: %s\\n", cam->projectionStr().c_str());
        sprintf(m+strlen(m), "Animation: %s\\n", cam->animationStr().c_str());
        sprintf(m+strlen(m), "Max speed: %4.1f/sec.\\n", cam->maxSpeed());
        if (camera()->projection() > P_monoOrthographic)
        {   SLfloat eyeSepPC = cam->eyeSeparation()/cam->focalDist()*100;
            sprintf(m+strlen(m), "Eye separation: %4.2f (%3.1f%% of f)\\n", cam->eyeSeparation(), eyeSepPC);
        }
        sprintf(m+strlen(m), "fov: %4.2f\\n", cam->fov());
        sprintf(m+strlen(m), "Focal distance (f): %4.2f\\n", cam->focalDist());
        sprintf(m+strlen(m), "Projection size: %4.2f x %4.2f\\n", cam->focalDistScrW(), cam->focalDistScrH());
    }

    if (_showStatsMemory)
    {
        // Calculate voxel contents
        SLfloat vox = (SLfloat)_stats.numVoxels;
        SLfloat voxEmpty = (SLfloat)_stats.numVoxEmpty;
        SLfloat voxelsEmpty  = vox ? voxEmpty / vox*100.0f : 0.0f;
        SLfloat numRTTria = (SLfloat)_stats.numTriangles;
        SLfloat avgTriPerVox = vox ? numRTTria / (vox-voxEmpty) : 0.0f;
        SLint numRenderedPC = (SLint)((SLfloat)cam->numRendered()/(SLfloat)_stats.numLeafNodes * 100.0f);

        // Calculate total size of texture bytes on CPU
        SLuint cpuTexMemoryBytes = 0;
        for (auto t : s->textures())
            for (auto i : t->images())
                cpuTexMemoryBytes += i->bytesPerImage();

        sprintf(m+strlen(m), "Memory -------------------------------------\\n");
        sprintf(m+strlen(m), "No. of Group/Leaf Nodes: %d / %d\\n", _stats.numGroupNodes,  _stats.numLeafNodes);
        sprintf(m+strlen(m), "Nodes in Frustum: %d (%d%%)\\n", cam->numRendered(), numRenderedPC);
        sprintf(m+strlen(m), "Lights: %d\\n", _stats.numLights);
        sprintf(m+strlen(m), "CPU MB in Tex.: %3.2f\\n", (SLfloat)cpuTexMemoryBytes / 1E6f);
        sprintf(m+strlen(m), "CPU MB in Meshes: %3.2f\\n", (SLfloat)_stats.numBytes / 1E6f);
        sprintf(m+strlen(m), "CPU MB in Voxel.: %3.2f\\n", (SLfloat)_stats.numBytesAccel / 1E6f);
        sprintf(m+strlen(m), "CPU MB in Total: %3.2f\\n", (SLfloat)(cpuTexMemoryBytes + _stats.numBytes + _stats.numBytesAccel) / 1E6f);
        sprintf(m+strlen(m), "GPU MB in VBO: %4.2f\\n", (SLfloat)SLGLVertexBuffer::totalBufferSize / 1E6f);
        sprintf(m+strlen(m), "GPU MB in Tex.: %4.2f\\n", (SLfloat)SLGLTexture::numBytesInTextures / 1E6f);
        sprintf(m+strlen(m), "GPU MB in Total: %3.2f\\n", (SLfloat)(SLGLVertexBuffer::totalBufferSize + SLGLTexture::numBytesInTextures) / 1E6f);
        sprintf(m+strlen(m), "No. of Voxels/empty: %d / %4.1f%%\\n", _stats.numVoxels, voxelsEmpty);
        sprintf(m+strlen(m), "Avg. & Max. Tria/Voxel: %4.1f / %d\\n", avgTriPerVox, _stats.numVoxMaxTria);
        sprintf(m+strlen(m), "Group & Leaf Nodes: %u / %u\\n", _stats.numGroupNodes, _stats.numLeafNodes);
        sprintf(m+strlen(m), "Meshes & Triangles: %u / %u\\n", _stats.numMeshes, _stats.numTriangles);
    }

    if (_showStatsVideo)
    {
        SLCVCalibration& cal = s->calibration();
        SLCVSize capSize = SLCVCapture::captureSize;
        SLVideoType vt = s->videoType();

        sprintf(m+strlen(m), "Video --------------------------------------\\n");
        sprintf(m+strlen(m), "Video Type: %s\\n", vt==0 ? "None" : vt==1 ? "Main Camera" : "Secondary Camera");
        sprintf(m+strlen(m), "Display size: %d x %d\\n", cal.imageSize().width, cal.imageSize().height);
        sprintf(m+strlen(m), "Capture size: %d x %d\\n", capSize.width, capSize.height);
        sprintf(m+strlen(m), "Field of view (deg.): %4.1f\\n", cal.cameraFovDeg());
        sprintf(m+strlen(m), "fx, fy, cx, cy: %4.1f,%4.1f,%4.1f,%4.1f\\n", cal.fx(),cal.fy(),cal.cx(),cal.cy());
        sprintf(m+strlen(m), "fx/Width: %4.2f\\n", cal.fx()/cal.imageSize().width);
        sprintf(m+strlen(m), "Calibration time: %s\\n", cal.calibrationTime().c_str());
    }

    SLTexFont* f = SLTexFont::getFont(1.2f, SL::dpi);
    SLText* t = new SLText(m, f, SLCol4f::WHITE, (SLfloat)_scrW, 1.0f);
    t->translate(5.0f, -t->size().y-5.0f, 0.0f, TS_object);
    s->infoGL(t);
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::build2DInfoRT builds the _infoRT group with info texts. 
See SLButton and SLText class for more infos. 
*/
void SLSceneView::build2DInfoRT()
{     
    SLScene* s = SLScene::current;
    if (s->infoRT()) 
        delete s->infoRT();
  
    // prepare some statistic infos
    SLCamera* cam = camera();
    SLRaytracer* rt = &_raytracer;
    SLint  primaries = _scrW * _scrH;
    SLuint total = primaries + SLRay::reflectedRays + SLRay::subsampledRays + SLRay::refractedRays + SLRay::shadowRays;
    SLfloat vox = (SLfloat)_stats.numVoxels;
    SLfloat voxEmpty = (SLfloat)_stats.numVoxEmpty;
    SLfloat voxelsEmpty  = vox ? voxEmpty / vox*100.0f : 0.0f;
    SLfloat numRTTria = (SLfloat)_stats.numTriangles;
    SLfloat avgTriPerVox = vox ? numRTTria / (vox-voxEmpty) : 0.0f;
    SLfloat rpms = rt->renderSec() ? total/rt->renderSec()/1000.0f : 0.0f;
   
    SLchar m[2550];   // message character array
    m[0]=0;           // set zero length

    if (_showStatsTiming)
    {
        sprintf(m+strlen(m), "Timing -------------------------------------\\n");
        sprintf(m+strlen(m), "Scene: %s\\n", s->name().c_str());
        sprintf(m+strlen(m), "Time per frame: %4.2f sec.  (Size: %d x %d)\\n", rt->renderSec(), _scrW, _scrH);
        sprintf(m+strlen(m), "fov: %4.2f\\n", cam->fov());
        sprintf(m+strlen(m), "Focal dist. (f): %4.2f\\n", cam->focalDist());
        sprintf(m+strlen(m), "Rays per millisecond: %6.0f\\n", rpms);
        sprintf(m+strlen(m), "Threads: %d\\n", rt->numThreads());
    }

    if (_showStatsRenderer)
    {
        sprintf(m+strlen(m), "Renderer -----------------------------------\\n");
        sprintf(m+strlen(m), "Max. allowed RT depth: %d\\n", SLRay::maxDepth);
        sprintf(m+strlen(m), "Max. reached RT depth: %d\\n", SLRay::maxDepthReached);
        sprintf(m+strlen(m), "Average RT depth: %4.2f\\n", SLRay::avgDepth/primaries);
        sprintf(m+strlen(m), "AA threshold: %2.1f\\n", rt->aaThreshold());
        sprintf(m+strlen(m), "AA samples: %d x %d\\n", rt->aaSamples(), rt->aaSamples());
        sprintf(m+strlen(m), "AA pixels: %u, %3.1f%%\\n", SLRay::subsampledPixels, (SLfloat)SLRay::subsampledPixels/primaries*100.0f);
        sprintf(m+strlen(m), "Primary rays: %u, %3.1f%%\\n", primaries, (SLfloat)primaries/total*100.0f);
        sprintf(m+strlen(m), "Reflected rays: %u, %3.1f%%\\n", SLRay::reflectedRays, (SLfloat)SLRay::reflectedRays/total*100.0f);
        sprintf(m+strlen(m), "Refracted rays: %u, %3.1f%%\\n", SLRay::refractedRays, (SLfloat)SLRay::refractedRays/total*100.0f);
        sprintf(m+strlen(m), "Ignored rays: %u, %3.1f%%\\n", SLRay::ignoredRays, (SLfloat)SLRay::ignoredRays/total*100.0f);
        sprintf(m+strlen(m), "TIR rays: %u, %3.1f%%\\n", SLRay::tirRays, (SLfloat)SLRay::tirRays/total*100.0f);
        sprintf(m+strlen(m), "Shadow rays: %u, %3.1f%%\\n", SLRay::shadowRays, (SLfloat)SLRay::shadowRays/total*100.0f);
        sprintf(m+strlen(m), "AA rays: %u, %3.1f%%\\n", SLRay::subsampledRays, (SLfloat)SLRay::subsampledRays/total*100.0f);
        sprintf(m+strlen(m), "Total rays: %u, %3.1f%%\\n", total, 100.0f);
        #ifdef _DEBUG
        sprintf(m+strlen(m), "Intersection tests: %u\\n", SLRay::intersections);
        sprintf(m+strlen(m), "Intersections: %u, %3.1f%%\\n", SLRay::tests, SLRay::intersections/(SLfloat)SLRay::tests*100.0f);
        #endif
    }

    if (_showStatsCamera)
    {
        sprintf(m+strlen(m), "Camera -------------------------------------\\n");
        sprintf(m+strlen(m), "Projection: %s\\n", cam->projectionStr().c_str());
        sprintf(m+strlen(m), "Animation: %s\\n", cam->animationStr().c_str());
        sprintf(m+strlen(m), "Max speed: %4.1f/sec.\\n", cam->maxSpeed());
        if (camera()->projection() > P_monoOrthographic)
        {   SLfloat eyeSepPC = cam->eyeSeparation()/cam->focalDist()*100;
            sprintf(m+strlen(m), "Eye separation: %4.2f (%3.1f%% of f)\\n", cam->eyeSeparation(), eyeSepPC);
        }
        sprintf(m+strlen(m), "fov: %4.2f\\n", cam->fov());
        sprintf(m+strlen(m), "Focal distance (f): %4.2f\\n", cam->focalDist());
        sprintf(m+strlen(m), "Projection size: %4.2f x %4.2f\\n", cam->focalDistScrW(), cam->focalDistScrH());
    }

    if (_showStatsMemory)
    {
        // Calculate total size of texture bytes on CPU
        SLuint cpuTexMemoryBytes = 0;
        for (auto t : s->textures())
            for (auto i : t->images())
                cpuTexMemoryBytes += i->bytesPerImage();

        sprintf(m+strlen(m), "Memory -------------------------------------\\n");
        sprintf(m+strlen(m), "Group Nodes: %d\\n", _stats.numGroupNodes);
        sprintf(m+strlen(m), "Leaf Nodes: %d\\n", _stats.numLeafNodes);
        sprintf(m+strlen(m), "Lights: %d\\n", _stats.numLights);
        sprintf(m+strlen(m), "CPU MB in Textures: %f\\n", (SLfloat)cpuTexMemoryBytes / 1000000.0f);
        sprintf(m+strlen(m), "CPU MB in Meshes: %f\\n", (SLfloat)_stats.numBytes / 1000000.0f);
        sprintf(m+strlen(m), "CPU MB in Voxel.: %f\\n", (SLfloat)_stats.numBytesAccel / 1000000.0f);
        sprintf(m+strlen(m), "CPU MB in Total: %f\\n", (SLfloat)(cpuTexMemoryBytes + _stats.numBytes + _stats.numBytesAccel) / 1000000.0f);
        sprintf(m+strlen(m), "Triangles: %d\\n", _stats.numTriangles);
        sprintf(m+strlen(m), "Voxels: %d\\n", _stats.numVoxels);
        sprintf(m+strlen(m), "Voxels empty: %4.1f%%\\n", voxelsEmpty);
        sprintf(m+strlen(m), "Avg. Tria./Voxel: %4.1f\\n", avgTriPerVox);
        sprintf(m+strlen(m), "Max. Tria./Voxel: %d", _stats.numVoxMaxTria);
    }

    SLTexFont* f = SLTexFont::getFont(1.2f, SL::dpi);
    SLText* t = new SLText(m, f, SLCol4f::WHITE, (SLfloat)_scrW, 1.0f);
    t->translate(10.0f, -t->size().y-5.0f, 0.0f, TS_object);
    s->infoRT(t);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::build2DInfoLoading builds the _infoLoading info texts.
See SLButton and SLText class for more infos.
*/
void SLSceneView::build2DInfoLoading()
{
    SLScene* s = SLScene::current;
    if (s->infoLoading()) return;
    SLTexFont* f = SLTexFont::getFont(3, SL::dpi);
    SLText* t = new SLText("Loading Scene . . .", f, SLCol4f::WHITE, (SLfloat)_scrW, 1.0f);
    t->translate(10.0f, -t->size().y-5.0f, 0.0f, TS_object);
    t->translate(_scrW*0.5f - t->size().x*0.5f, -(_scrH*0.5f) + t->size().y, 0.0f, TS_object);
    s->infoLoading(t);
}
//-----------------------------------------------------------------------------
/*!
Builds the message buttons. They depend on screen width.
*/
void SLSceneView::build2DMsgBoxes()
{ 
    SLScene*    s = SLScene::current;
    SLTexFont*  f = SLTexFont::getFont(1.7f, SL::dpi);
   
    // Help button
    if (s->btnHelp()) delete s->btnHelp();
    s->btnHelp(new SLButton(this, s->infoHelp_en(), f,
                            C_aboutToggle, false, false, 0, true,
                            _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                            SLCol3f::COLBFH, 0.8f, TA_centerCenter));

    _stateGL->modelViewMatrix.identity();
    s->btnHelp()->drawBits()->off(SL_DB_HIDDEN);
    s->btnHelp()->setSizeRec();
    s->btnHelp()->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->btnHelp()->updateAABBRec();
   
    // About button
    if (s->btnAbout()) delete s->btnAbout();
    s->btnAbout(new SLButton(this, s->infoAbout_en(), f,
                             C_aboutToggle, false, false, 0, true,
                             _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                             SLCol3f::COLBFH, 0.8f, TA_centerCenter));

    _stateGL->modelViewMatrix.identity();
    s->btnAbout()->drawBits()->off(SL_DB_HIDDEN);
    s->btnAbout()->setSizeRec();
    s->btnAbout()->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->btnAbout()->updateAABBRec();
   
    // Credits button
    if (s->btnCredits()) delete s->btnCredits();
    s->btnCredits(new SLButton(this, s->infoCredits_en(), f,
                               C_aboutToggle, false, false, 0, true,
                               _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                               SLCol3f::COLBFH, 0.8f, TA_centerCenter));

    _stateGL->modelViewMatrix.identity();
    s->btnCredits()->drawBits()->off(SL_DB_HIDDEN);
    s->btnCredits()->setSizeRec();
    s->btnCredits()->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->btnCredits()->updateAABBRec();
   
    // No calibration button
    if (s->btnNoCalib()) delete s->btnNoCalib();
    s->btnNoCalib(new SLButton(this, s->infoNoCalib_en(), f,
                               C_noCalibToggle, false, false, 0, true,
                               _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                               SLCol3f::COLBFH, 0.8f, TA_centerCenter));

    _stateGL->modelViewMatrix.identity();
    s->btnNoCalib()->drawBits()->off(SL_DB_HIDDEN);
    s->btnNoCalib()->setSizeRec();
    s->btnNoCalib()->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->btnNoCalib()->updateAABBRec();
}
//-----------------------------------------------------------------------------





//-----------------------------------------------------------------------------
/*!
Returns the window title with name & FPS
*/
SLstring SLSceneView::windowTitle()
{  
    SLScene* s = SLScene::current;
    SLchar title[255];

    if (_renderType == RT_rt)
    {   if (_raytracer.continuous())
        {   sprintf(title, "%s (fps: %4.1f, Threads: %d)", 
                    s->name().c_str(), 
                    s->fps(),
                    _raytracer.numThreads());
        } else
        {   sprintf(title, "%s (%d%%, Threads: %d)", 
                    s->name().c_str(), 
                    _raytracer.pcRendered(), 
                    _raytracer.numThreads());
        }
    } else
    if (_renderType == RT_pt)
    {   sprintf(title, "%s (%d%%, Threads: %d)", 
                s->name().c_str(), 
                _pathtracer.pcRendered(), 
                _pathtracer.numThreads());
    } else
    {   SLuint nr = _camera->numRendered() ? _camera->numRendered() : _stats.numNodes;
        if (s->fps() > 5)
            sprintf(title, "%s (fps: %4.0f, %u nodes of %u rendered)",
                    s->name().c_str(), s->fps(), nr, _stats.numNodes);
        else
            sprintf(title, "%s (fps: %4.1f, %u nodes of %u rendered)",
                    s->name().c_str(), s->fps(), nr, _stats.numNodes);
    }
    return SLstring(title);
}
//-----------------------------------------------------------------------------
/*!
Starts the ray tracing & sets the RT menu
*/
void SLSceneView::startRaytracing(SLint maxDepth)
{  
    SLScene* s = SLScene::current;
    _renderType = RT_rt;
    _stopRT = false;
    _raytracer.maxDepth(maxDepth);
    _raytracer.aaSamples(_doMultiSampling?3:1);
    s->menu2D(s->menuRT());
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
    if (_raytracer.state()==rtReady)
    {
        SLScene* s = SLScene::current;

        // Update transforms and aabbs
        // @Todo: causes multithreading bug in RT
        //s->root3D()->needUpdate();

        // Do software skinning on all changed skeletons
        for (auto mesh : s->meshes())
            mesh->updateAccelStruct();

        // Start raytracing
        if (_raytracer.distributed())
             _raytracer.renderDistrib(this);
        else _raytracer.renderClassic(this);
    }

    // Refresh the render image during RT
    _raytracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if(_stopRT)
    {   _renderType = RT_gl;
        SLScene* s = SLScene::current;
        s->menu2D(s->menuGL());
        s->menu2D()->closeAll();
        updated = true;
    }

    return updated;
}
//-----------------------------------------------------------------------------
/*!
Starts the pathtracing
*/
void SLSceneView::startPathtracing(SLint maxDepth, SLint samples)
{  
    SLScene* s = SLScene::current;
    _renderType = RT_pt;
    _stopPT = false;
    _pathtracer.maxDepth(maxDepth);
    _pathtracer.aaSamples(samples);
    s->menu2D(s->menuPT());
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
    if (_pathtracer.state()==rtReady)
    {
        SLScene* s = SLScene::current;

        // Update transforms and AABBs
        s->root3D()->needUpdate();

        // Do software skinning on all changed skeletons
        for (auto mesh : s->meshes())
            mesh->updateAccelStruct();

        // Start raytracing
        _pathtracer.render(this);
    }

    // Refresh the render image during PT
    _pathtracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if(_stopPT)
    {   _renderType = RT_gl;
        SLScene* s = SLScene::current;
        s->menu2D(s->menuGL());
        s->menu2D()->closeAll();
        updated = true;
    }

    return updated;
}
//-----------------------------------------------------------------------------
//! Sets the loading text flags
void SLSceneView::showLoading(SLbool showLoading) 
{
    if (showLoading)
    {
        if (!_stateGL)
        {   // This can happen if show loading is called before a new scene is set
            SLScene* s = SLScene::current;
            _stateGL = SLGLState::getInstance();
            _stateGL->onInitialize(s->background().colors()[0]);
        }
        _stateGL->clearColor(SLCol4f::GRAY);
        _stateGL->clearColorDepthBuffer();
    }
    _showLoading = showLoading;
}
//------------------------------------------------------------------------------
//! Handles the test setting
SLbool SLSceneView::testRunIsFinished()
{
    if (SL::testFrameCounter == 0)
        SLScene::current->timerStart();

    if (SLScene::current->timeSec() > SL::testDurationSec)
    {   
        if (SL::testScene==C_sceneAll)
        {   if (SL::testSceneAll < C_sceneRTTest)
            {   
                SLfloat fps = (SLfloat)SL::testFrameCounter / (SLfloat)SL::testDurationSec;
                SL_LOG("%s: Frames: %5u, FPS=%6.1f\n", 
                       SL::testSceneNames[SL::testSceneAll].c_str(), 
                       SL::testFrameCounter, 
                       fps);
                
                // Start next scene
                SL::testFrameCounter = 0;
                SL::testSceneAll = (SLCommand)(SL::testSceneAll + 1);
                if (SL::testSceneAll == C_sceneLargeModel)
                    SL::testSceneAll = (SLCommand)(SL::testSceneAll + 1);
                onCommand(SL::testSceneAll);
                SLScene::current->timerStart();
            } else
            {   
                SL_LOG("------------------------------------------------------------------\n");
                onCommand(C_quit);
                return true;
            }
        } else
        {   onCommand(C_quit);
            return true;
        }
    }
    SL::testFrameCounter++;

    return false;
}
//------------------------------------------------------------------------------
