//#############################################################################
//  File:      SLSceneView.cpp
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLLight.h>
#include <SLCamera.h>
#include <SLAABBox.h>
#include <SLGLShaderProg.h>
#include <SLAnimation.h>
#include <SLLightSphere.h>
#include <SLLightRect.h>
#include <SLRay.h>
#include <SLTexFont.h>
#include <SLButton.h>
#include <SLBox.h>

SLfloat SLSceneView::_lastFrameMS = 0.0f;

//-----------------------------------------------------------------------------
//! SLSceneView default constructor
/*! The default constructor adds the this pointer to the sceneView vector in 
SLScene. If an inbetween element in the vector is zero (from previous sceneviews) 
it will be replaced. The sceneviews _index is the index in the sceneview vector.
It never changes throughout the life of a sceneview. 
*/
SLSceneView::SLSceneView() : SLObject()
{ 
    SLScene* s = SLScene::current;
    assert(s && "No SLScene::current instance.");
   
    for (SLint i=0; i<s->_sceneViews.size(); ++i)
    {  if (s->_sceneViews[i]==0)
        {   s->_sceneViews[i] = this;
            _index = i;
            return;
        }
    }
   
    // No gaps, so add it and get the index back.
    s->_sceneViews.push_back(this);
    _index = (SLuint)s->_sceneViews.size() - 1;
}
//-----------------------------------------------------------------------------
SLSceneView::~SLSceneView()
{  
    // Set pointer in SLScene::sceneViews vector to zero but leave it.
    SLScene::current->_sceneViews[_index] = 0;
    SL_LOG("~SLSceneView\n");
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::init initializes default values for an empty scene
*/
void SLSceneView::init(SLstring name, 
                       SLint screenWidth, 
                       SLint screenHeight, 
                       SLint dotsPerInch, 
                       SLVstring& cmdLineArgs,
                       void* onWndUpdateCallback,
                       void* onSelectNodeMeshCallback,
                       void* onShowSystemCursorCallback)
{  
    _name = name;
    _scrW = screenWidth;
    _scrH = screenHeight;
    _dpi = dotsPerInch;
	_vrMode = false;

    /* The window update callback function is used to refresh the ray tracing 
    image during the rendering process. The ray tracing image is drawn by OpenGL 
    as a texture on a single quad.*/
    onWndUpdate = (cbOnWndUpdate)onWndUpdateCallback;

    /* The on select node callback is called when a node got selected on double
    click, so that the UI can react on it.*/
    onSelectedNodeMesh = (cbOnSelectNodeMesh) onSelectNodeMeshCallback;

    /* We need access to the system specific cursor and be able to hide it
    if we need to draw our own. 
    @todo could be simplified if we implemented our own SLWindow class */
    onShowSysCursor = (cbOnShowSysCursor) onShowSystemCursorCallback;

    _stateGL = 0;
   
    _camera = 0;
   
    // enables and modes
    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;
    _touchDowns = 0;

    _doDepthTest = true;
    _doMultiSampling = true;    // true=OpenGL multisampling is turned on
    _hasMultiSampling = false;  // Multisampling is check in onInitialize
    _doFrustumCulling = true;   // true=enables view frustum culling
    _waitEvents = true;
    _drawBits.allOff();
       
    _stats.clear();
    _showMenu = true;
    _showStats = false;
    _showInfo = true;
    _showLoading = false;

    _fps = 0.0f,               // Frames per second
    _scrWdiv2 = _scrW>>1;
    _scrHdiv2 = _scrH>>1;
    _scrWdivH = (SLfloat)_scrW / (SLfloat)_scrH;
      
    _renderType = renderGL;

    onStartup(cmdLineArgs); 
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onInitialize is called by the window system before the first 
rendering. It applies all scene rendering attributes with the according 
OpenGL function.
*/
void SLSceneView::initSceneViewCamera(const SLVec3f& dir, SLProjection proj)
{             
    _sceneViewCamera.camAnim(turntableYUp);
    _sceneViewCamera.name("SceneViewCamera");
    _sceneViewCamera.clipNear(.1f);
    _sceneViewCamera.clipFar(2000.0f);
    _sceneViewCamera.speedLimit(40);
    _sceneViewCamera.eyeSeparation(_sceneViewCamera.focalDist()/30.0f);
    _sceneViewCamera.setProjection(this, centerEye);
  
	// ignore projection if in vr mode
	if(!_vrMode)
		_sceneViewCamera.projection(proj);

    // fit scenes bounding box in view frustum
    SLScene* s = SLScene::current;
    if (s->root3D())
    {
        // we want to fit the scenes combined aabb in the view frustum
        SLAABBox* sceneBounds = s->root3D()->aabb();

        _sceneViewCamera.position(sceneBounds->centerWS(), TS_World);
        _sceneViewCamera.lookAt(sceneBounds->centerWS() + dir, SLVec3f::AXISY, TS_Parent);

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

        /*
        // visualize the view space oriented bounding box
        SLMaterial* mat = new SLMaterial("test", SLCol4f::WHITE, SLCol4f::WHITE, 100.0f, 0.5f, 0.5f, 0.5f);
        mat->translucency(0.5f);
        mat->transmission(SLCol4f(0, 0, 0, 0.5f));
        SLBox* box = new SLBox(-1, -1, -1, 1, 1, 1, "test", mat);
        
        SLNode* testNode = new SLNode;
        testNode->scale(vsMax.x, vsMax.y, vsMax.z);
        SLMat4f vsRot = _sceneViewCamera.updateAndGetWM();
        vsRot.translation(0, 0, 0);
        testNode->multiply(vsRot);
        testNode->translation(sceneBounds->centerWS());
        testNode->addMesh(box);
        testNode->buildAABBRec();
        s->root3D()->addChild(testNode);
         */
        
        SLfloat dist = 0.0f;
        SLfloat distX = 0.0f;
        SLfloat distY = 0.0f;
        SLfloat halfTan = tan(SL_DEG2RAD*_sceneViewCamera.fov()*0.5f);

        // @todo There is still a bug when OSX doesn't pass correct GLWidget size
        // correctly set the camera distance...
        SLfloat ar = _sceneViewCamera.aspect();

        // special case for orthographic cameras
        if (proj == monoOrthographic)
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
        _sceneViewCamera.translate(SLVec3f(0, 0, dist), TS_Local);
    }

    _stateGL->modelViewMatrix.identity();
    _sceneViewCamera.updateAABBRec();

	// if no camera exists or in VR mode use the sceneViewCamera
	if(_camera == 0 || _vrMode)
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
        _sceneViewCamera.position(position);
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
    _stateGL->onInitialize(s->backColor());
    
    _blendNodes.clear();
    _opaqueNodes.clear();

    _raytracer.clearData();
    _renderType = renderGL;

    _fps = 0;         
    _frameTimeMS.init(); 
    _updateTimeMS.init();
    _cullTimeMS.init();  
    _draw3DTimeMS.init();
    _draw2DTimeMS.init();



    // Check for multisample capability
    SLint samples;
    glGetIntegerv(GL_SAMPLES, &samples);
    _hasMultiSampling = (samples > 0);

    // init 3D scene with initial depth 1
    if (s->root3D() && s->root3D()->aabb()->radiusOS()==0)
    {
        // build axis aligned bounding box hierarchy after init
        clock_t t = clock();
        s->_root3D->updateAABBRec();
      
        SL_LOG("Time for AABBs : %5.3f sec.\n", 
                (SLfloat)(clock()-t)/(SLfloat)CLOCKS_PER_SEC);
      
        s->root3D()->statsRec(_stats);
    }

    initSceneViewCamera();
   
    build2DMenus();
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onResize is called by the window system before the first 
rendering and whenever the window changes its size.
*/
void SLSceneView::onResize(const SLint width, const SLint height)
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
        if (_camera && _camera->projection() == stereoSideBySideD)
        {
            _oculusFB.updateSize((SLint)(s->oculus()->resolutionScale()*(SLfloat)_scrW), 
                                 (SLint)(s->oculus()->resolutionScale()*(SLfloat)_scrH));
            s->oculus()->renderResolution(_scrW, _scrH);
        }
      
        // Stop raytracing & pathtracgin on resize
        if (_renderType != renderGL)
        {   _renderType = renderGL;
            _raytracer.continuous(false);
            s->menu2D(s->_menuGL);
            s->menu2D()->hideAndReleaseRec();
            s->menu2D()->drawBits()->off(SL_DB_HIDDEN);
        }
    }
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onPaint is called by window system whenever the window therefore 
the scene needs to be painted. Depending on the renderer it calls first
SLSceneView::updateAndDrawGL3D, SLSceneView::updateAndDrawRT3D or
SLSceneView::updateAndDrawPT3D and then SLSceneView::updateAndDraw2D for all
UI in 2D. The method returns true if either the 2D or 3D graph was updated or 
waitEvents is false.
*/
SLbool SLSceneView::onPaint()
{  
    SLScene* s = SLScene::current;

    // calculate elapsed time

    // @todo VERY IMPORTANT this is an other instance where we would need a central parent
    //       class to the scene view. the delta time should be per time step in the scene.
    //       Inbetween those time steps the parent class does the animation update etc.
    //       and the scene views don't need the deltaTime anymore but are merely what they
    //       should be, which is displays for the current state of the scene.
    //       (of course a scene view would need the delta time if it wanted to move an object)
    SLfloat timeNowMS = s->timeMilliSec();
    SLfloat elapsedTimeMS = timeNowMS - _lastFrameMS;
    SLGLBuffer::totalDrawCalls = 0;
    
    SLbool updated3D = false;
   
    if (_camera  && s->_root3D)
    {   // Render the 3D scenegraph by by raytracing, pathtracing or OpenGL
        switch (_renderType)
        {   case renderRT: updated3D = updateAndDrawRT3D(elapsedTimeMS); break;
            case renderPT: updated3D = updateAndDrawPT3D(elapsedTimeMS); break;
            default:       updated3D = updateAndDrawGL3D(elapsedTimeMS); break;
        }
    };

    // Render the 2D GUI (menu etc.)
    SLfloat startMS = s->timeMilliSec();
    SLbool updated2D = updateAndDraw2D(elapsedTimeMS);
    _draw2DTimeMS.set(s->timeMilliSec()-startMS);
    _stateGL->unbindAnythingAndFlush();

    if (_camera->projection() == stereoSideBySideD)
        s->oculus()->endFrame(_scrW, _scrH, _oculusFB.texID());

    // Calculate the frames per second metric
    calcFPS(elapsedTimeMS);
    _lastFrameMS = timeNowMS;

    // Update statisitc of VBO's & drawcalls
    _totalBufferCount = SLGLBuffer::totalBufferCount;
    _totalBufferSize = SLGLBuffer::totalBufferSize;
    _totalDrawCalls = SLGLBuffer::totalDrawCalls;
    SLGLBuffer::totalDrawCalls   = 0;

    // Return true if a repaint is needed
    return !_waitEvents || updated3D || updated2D;
}




//-----------------------------------------------------------------------------
//! CompareNodeViewDist C-function declaration to avoid XCode warning
SLbool CompareNodeViewDist(SLNode* a, SLNode* b); 
//-----------------------------------------------------------------------------
/*! 
CompareNodeViewDist C-function serves as the sort comparison function for the 
blend sorting.
*/
SLbool CompareNodeViewDist(SLNode* a, SLNode* b)
{   if (!a) return false;
    if (!b) return true;
    return a->aabb()->sqrViewDist() > b->aabb()->sqrViewDist();
}
//-----------------------------------------------------------------------------
//! Draws the 3D scene with OpenGL
/*! This is main routine for updating and drawing the 3D scene for one frame. 
The following steps are processed:
<ol>
<li>
<b>Animate and Update Scenegraph</b>:
For all nodes that have an animation attached a new local transform is 
calculated and its flag SLNode::_isWMUpToDate becomes false. On each call of
SLSceneView::updateAndGetWM it is tested if the world matrix needs to be
updated because of a change on the local transform.
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
The frustum culling traversal fills the vectors SLSceneView::_opaqueNodes 
and SLSceneView::_blendNodes with the visible nodes. Nodes that are not
visible with the current camera are not drawn. 
</li>
<li>
<b>Draw Opaque and Blended Nodes</b>:
By calling the SLSceneView::draw3D all nodes in the vectors 
SLSceneView::_opaqueNodes and SLSceneView::_blendNodes will be drawn.
If a stereo projection is set, the scene gets drawn a second time for
the right eye.
</li>
<li>
<b>Draw Oculus Framebuffer</b>:
If the projection is the Oculus stereo projection the frambuffer image
is drawn.
</li>
</ol>
*/
SLbool SLSceneView::updateAndDrawGL3D(SLfloat elapsedTimeMS)
{
    SLScene* s = SLScene::current;

    preDraw();
    
    ////////////////////////////////////
    // 1. Animate & Update Scenegraph //
    ////////////////////////////////////

    // Do animations
    SLfloat startMS = s->timeMilliSec();
    SLbool animated = !drawBit(SL_DB_ANIMOFF) &&
                      !s->_root3D->drawBits()->get(SL_DB_ANIMOFF) &&
                       s->_root3D->animateRec(elapsedTimeMS);

    // don't slow down if we're in HMD stereo mode
    animated = animated || _camera->projection() == stereoSideBySideD;

    // Update camera animation seperately to process input on camera object
    SLbool camUpdated = _camera->camUpdate(elapsedTimeMS);
   
    // Update the world matrix & AABBs efficiently
    _stateGL->modelViewMatrix.identity();
    s->_root3D->updateAABBRec();

    _updateTimeMS.set(s->timeMilliSec()-startMS);


    //////////////////////
    // 2. Clear Buffers //
    //////////////////////
    
    // Render into framebuffer if Oculus stereo projection is used
    if (_camera->projection() == stereoSideBySideD)
    {   s->oculus()->beginFrame();
        
        // @todo change SLGLOculusFB to not require a resolution here
        _oculusFB.bindFramebuffer((SLint)(s->oculus()->resolutionScale() * (SLfloat)_scrW), 
                                  (SLint)(s->oculus()->resolutionScale() * (SLfloat)_scrH)); 
    }

    // Clear buffers
    _stateGL->clearColor(s->backColor());
    _stateGL->clearColorDepthBuffer();

    // Change state (only when changed)
    _stateGL->multiSample(_doMultiSampling);
    _stateGL->depthTest(_doDepthTest);
    

    //////////////////////////////
    // 3. Set Projection & View //
    //////////////////////////////

    startMS = s->timeMilliSec();

    // Set projection and viewport
    if (_camera->projection() > monoOrthographic)   
         _camera->setProjection(this, leftEye);
    else _camera->setProjection(this, centerEye);

    // Set view center eye or left eye
    if (_camera->projection() > monoOrthographic)   
         _camera->setView(this, leftEye);
    else _camera->setView(this, centerEye);

    ////////////////////////
    // 4. Frustum Culling //
    ////////////////////////
   
    // Do frustum culling
    _camera->setFrustumPlanes(); 
    _blendNodes.clear();
    _opaqueNodes.clear();
    s->_root3D->cullRec(this);        
    if (!_doFrustumCulling) 
        _camera->numRendered(_stats.numLeafNodes);
   
    _cullTimeMS.set(s->timeMilliSec()-startMS);


    ////////////////////////////////////
    // 5. Draw Opaque & Blended Nodes //
    ////////////////////////////////////

    startMS = s->timeMilliSec();

    // We could also draw the scenegraph recursively
    // but this doesn't split transparent from opaque nodes
    //s->_root3D->drawRec(this);

    draw3D();
   
    // For stereo draw for right eye
    if (_camera->projection() > monoOrthographic)   
    {   _camera->setProjection(this, rightEye);
        _camera->setView(this, rightEye);
        draw3D();
      
        // Enable all color channels again
        _stateGL->colorMask(1, 1, 1, 1); 
    }
 
    _draw3DTimeMS.set(s->timeMilliSec()-startMS);

    ////////////////////////////////
    // 6. Draw Oculus framebuffer //
    ////////////////////////////////

    // Render framebuffer if Oculus stereo projection is used
    if (_camera->projection() == stereoSideBySideD) 
    {
        // temporary oculus mesh visualization
        //s->oculus()->endFrame(_scrW, _scrH, _oculusFB.texID());
    }


    postDraw();

    GET_GL_ERROR; // Check if any OGL errors occured
    return animated || camUpdated;
}
//----------------------------------------------------------------------------- 
/*!
SLSceneView::draw3D2 renders the opaque nodes before blended nodes.
Opaque nodes must be drawn before the blended, transparent nodes.
During the cull traversal all nodes with opaque materials are flagged and 
added the to the array _opaqueNodes.
*/
void SLSceneView::draw3D()
{  
    // Render first the opaque shapes and their helper lines 
    _stateGL->blend(false);
    _stateGL->depthMask(true);

    draw3DNodeLines(_opaqueNodes);
    draw3DNodes(_opaqueNodes);

    // Render the helper lines of the blended shapes non-blended!
    draw3DNodeLines(_blendNodes);

    _stateGL->blend(true);
    _stateGL->depthMask(false);

    // Blended shapes must be sorted back to front
    std::sort(_blendNodes.begin(), _blendNodes.end(), CompareNodeViewDist);

    draw3DNodes(_blendNodes);

    // Blending must be turned off again for correct anyglyph stereo modes
    _stateGL->blend(false);
    _stateGL->depthMask(true);
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DNodeLines draws the AABB, the axis and the animation
curves from the passed node vector directly with their world coordinates after 
the view transform. The lines must be drawn without blending.
Colors:
Red   : AABB of nodes with meshes
Pink  : AABB of nodes without meshes (only child nodes)
Yellow: AABB of selected node 
*/
void SLSceneView::draw3DNodeLines(SLVNode &nodes)
{  
    // draw the opaque shapes directly w. their wm transform
    for(SLuint i=0; i<nodes.size(); ++i)
    {
        if (nodes[i] != _camera)
        {
            // Set the view transform
            _stateGL->modelViewMatrix.setMatrix(_stateGL->viewMatrix);

            // Draw first AABB of the shapes but not the camera
            if ((drawBit(SL_DB_BBOX) || nodes[i]->drawBit(SL_DB_BBOX)) &&
                !nodes[i]->drawBit(SL_DB_SELECTED))
            {
                if (nodes[i]->numMeshes() > 0)
                     nodes[i]->aabb()->drawWS(SLCol3f(1,0,0));
                else nodes[i]->aabb()->drawWS(SLCol3f(1,0,1));
            }

            // Draw AABB for selected shapes
            if (nodes[i]->drawBit(SL_DB_SELECTED))
                nodes[i]->aabb()->drawWS(SLCol3f(1,1,0));
      
            // Draw axis & animation curves
            if (drawBit(SL_DB_AXIS) || nodes[i]->drawBit(SL_DB_AXIS))
            {  
                nodes[i]->aabb()->drawAxisWS();

                // Draw the animation curve
                if (nodes[i]->animation())
                    nodes[i]->animation()->drawWS();
            }
        }
    }
   
    GET_GL_ERROR;        // Check if any OGL errors occured
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::draw3DNodes draws the nodes meshes from the passed node vector 
directly with their world coordinates after the view transform.
*/
void SLSceneView::draw3DNodes(SLVNode &nodes)
{  
    // draw the shapes directly with their wm transform
    for(SLuint i=0; i<nodes.size(); ++i)
    {
        // Set the view transform
        _stateGL->modelViewMatrix.setMatrix(_stateGL->viewMatrix);
      
        // Apply world transform
        _stateGL->modelViewMatrix.multiply(nodes[i]->updateAndGetWM().m());
      
        // Finally the nodes meshes
        nodes[i]->drawMeshes(this);
    }
   
    GET_GL_ERROR;  // Check if any OGL errors occured
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::updateAndDraw2D draws GUI tree in ortho projection. So far no
update is done to the 2D scenegraph.
*/
SLbool SLSceneView::updateAndDraw2D(SLfloat elapsedTimeMS)
{
    if (!_showMenu &&
        !_showInfo &&
        !_showLoading &&
        !_showStats &&
        _touchDowns==0 &&
        !_mouseDownL &&
        !_mouseDownM) return false;
    
    SLScene* s = SLScene::current;
    SLfloat w2 = (SLfloat)_scrWdiv2;
    SLfloat h2 = (SLfloat)_scrHdiv2;
    SLfloat depth = 0.9f;               // Render depth between -1 & 1

   
    // Set orthographic projection with 0,0,0 in the screen center
    // for now we just have one special gui case for side by side HMD stero rendering
    if (_camera->projection() != stereoSideBySideD)
    {        
        // @todo this doesn't need to be done every frame, we can save the current ortho matrix and update on resize
        _stateGL->projectionMatrix.ortho(-w2, w2,-h2, h2, 1.0f, -1.0f);
        // Set viewport over entire screen
        _stateGL->viewport(0, 0, _scrW, _scrH);

        draw2D();
    }
    else
    {
        // left eye
        _stateGL->projectionMatrix.setMatrix(s->oculus()->orthoProjection(leftEye));
        // Set viewport over entire screen
        _stateGL->viewport(0, 0, _oculusFB.halfWidth(), _oculusFB.height());
        
        draw2D();

        // left eye
        _stateGL->projectionMatrix.setMatrix(s->oculus()->orthoProjection(rightEye));
        // Set viewport over entire screen
        _stateGL->viewport(_oculusFB.halfWidth(), 0, _oculusFB.halfWidth(), _oculusFB.height());
        
        draw2D();
        
        // temp visualization of the texture above
        if (false)
        {
            glClear(GL_COLOR_BUFFER_BIT);
            static SLGLShaderProgGeneric tmpShader("StereoOculus.vert", "StereoOculus.frag");

            static GLuint screenQuad = 0;
            if (!screenQuad) {
                GLfloat quadVerts[] = {-1, -1,
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

            //bind the rift shader
            tmpShader.beginUse();

            glEnableVertexAttribArray(0);

            glBindBuffer(GL_ARRAY_BUFFER, screenQuad);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            glDisableVertexAttribArray(0);

            glEnable(GL_DEPTH_TEST);
        }

    }

    // below is the normal menu to test interaction with the default mouse
    if(false)
    {
        _stateGL->projectionMatrix.ortho(-w2, w2,-h2, h2, 1.0f, -1.0f);
        _stateGL->viewport(0, 0, _scrW, _scrH);

   
        _stateGL->depthMask(false);         // Freeze depth buffer for blending
        _stateGL->depthTest(false);         // Disable depth testing
        _stateGL->blend(true);              // Enable blending
        _stateGL->polygonLine(false);       // Only filled polygons

        // Draw menu buttons tree
        if (!_showLoading && _showMenu && s->_menu2D)
        {  _stateGL->modelViewMatrix.identity();
            _stateGL->modelViewMatrix.translate(-w2, -h2, 0);
            s->_menu2D->drawRec(this);
        }   
        _stateGL->blend(false);       // turn off blending
        _stateGL->depthMask(true);    // enable depth buffer writing
        _stateGL->depthTest(true);    // enable depth testing
        GET_GL_ERROR;                 // check if any OGL errors occured
    }
    
   return false;
}

//-----------------------------------------------------------------------------
/*!
SLSceneView::draw2D draws GUI tree in ortho projection. 
*/
void SLSceneView::draw2D()
{
    SLScene* s = SLScene::current;
    SLfloat w2 = (SLfloat)_scrWdiv2;
    SLfloat h2 = (SLfloat)_scrHdiv2;
    SLfloat depth = 1.0f;               // Render depth between -1 & 1

    _stateGL->pushModelViewMatrix();
    _stateGL->modelViewMatrix.identity();

    if (_camera->projection() == stereoSideBySideD)
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
        if (s->_infoLoading)
        {   _stateGL->pushModelViewMatrix();            
            _stateGL->modelViewMatrix.translate(-w2, h2, depth);
            _stateGL->modelViewMatrix.multiply(s->_infoLoading->om());
            s->_infoLoading->drawRec(this);
            _stateGL->popModelViewMatrix();
        }
    }

    // Draw statistics for GL
    if (!_showLoading && _showStats &&
        (s->_menu2D==s->_menuGL || s->_menu2D==s->_btnAbout))
    {   build2DInfoGL();
        if (s->_infoGL)
		{   _stateGL->pushModelViewMatrix();
			_stateGL->modelViewMatrix.translate(-w2, h2, depth);
			_stateGL->modelViewMatrix.translate(SLButton::minMenuPos.x, -SLButton::minMenuPos.y, 0);
            _stateGL->modelViewMatrix.multiply(s->_infoGL->om());
            s->_infoGL->drawRec(this);
            _stateGL->popModelViewMatrix();
        }
    }
   
    // Draw statistics for RT
    if (!_showLoading && _showStats &&
        (s->_menu2D==s->_menuRT))
    {   build2DInfoRT();
        if (s->_infoRT) 
        {   _stateGL->pushModelViewMatrix();  
            _stateGL->modelViewMatrix.translate(-w2, h2, depth);
            _stateGL->modelViewMatrix.multiply(s->_infoRT->om());
            s->_infoRT->drawRec(this);
            _stateGL->popModelViewMatrix();
        } 
    }

    // Draw scene info text if menuGL or menuRT is closed
    if (!_showLoading && 
        _showInfo && s->_info && 
        _camera->projection()<=monoOrthographic &&
        (s->_menu2D==s->_menuGL || 
         s->_menu2D==s->_menuRT ||
         s->_menu2D==s->_menuPT) && SLButton::buttonParent==0)
    {
        _stateGL->pushModelViewMatrix();  
        _stateGL->modelViewMatrix.translate(-w2, -h2, depth);
        _stateGL->modelViewMatrix.multiply(s->_info->om());
        s->_info->drawRec(this);
        _stateGL->popModelViewMatrix();
    }
   
    // Draw menu buttons tree
    if (!_showLoading && _showMenu && s->_menu2D)
    {   _stateGL->pushModelViewMatrix();  
        _stateGL->modelViewMatrix.translate(-w2, -h2, 0);
        
        s->_menu2D->drawRec(this);
        _stateGL->popModelViewMatrix();
    }   
   
    // 2D finger touch points  
    #ifndef SL_GLES2
    if (_touchDowns)
    {   _stateGL->multiSample(true);
        _stateGL->pushModelViewMatrix();  
      
        // Go to lower-left screen corner
        _stateGL->modelViewMatrix.translate(-w2, -h2, depth);
      
        SLVec3f* touch = new SLVec3f[_touchDowns];
        for (SLint i=0; i<_touchDowns; ++i)
        {   touch[i].x = (SLfloat)_touch[i].x;
            touch[i].y = (SLfloat)(_scrH - _touch[i].y);
            touch[i].z = 0.0f;
        }
      
        _bufTouch.generate(touch, _touchDowns, 3);
        delete [] touch;
      
        SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);
        _bufTouch.drawArrayAsConstantColorPoints(yelloAlpha, 21);
        _stateGL->popModelViewMatrix();
    }
    #endif

    // Draw turntable rotation crosshair
    if ((_mouseDownL || _mouseDownM) && _touchDowns==0)
    {   if (_camera->camAnim()==turntableYUp || _camera->camAnim()==turntableZUp)
        {   _stateGL->multiSample(true);
            _stateGL->pushModelViewMatrix();  
            _stateGL->modelViewMatrix.translate(0, 0, depth);
            SLVec3f cross;
            cross.set(0,0,0);
            _bufTouch.generate(&cross, 1, 3);
            SLCol4f yelloAlpha(1.0f, 1.0f, 0.0f, 0.5f);
            _bufTouch.drawArrayAsConstantColorPoints(yelloAlpha, (SLfloat)_dpi/12.0f);
            _stateGL->popModelViewMatrix();
        }
    }

    // Draw virtual mouse cursor if we're in hmd stereo mode
    if (_camera->projection() == stereoSideBySideD)
    {
        if (_camera->camAnim()==turntableYUp || _camera->camAnim()==turntableZUp)
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
    }

    _stateGL->popModelViewMatrix();        

    _stateGL->blend(false);       // turn off blending
    _stateGL->depthMask(true);    // enable depth buffer writing
    _stateGL->depthTest(true);    // enable depth testing
    GET_GL_ERROR;                 // check if any OGL errors occured
}

//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseDown gets called whenever a mouse button gets pressed and
dispatches the event to the currently attached eventhandler object.
*/
SLbool SLSceneView::onMouseDown(const SLMouseButton button, 
                                SLint x, SLint y, const SLKey mod)
{
    SLScene* s = SLScene::current;
   
    // Check first if mouse down was on a button    
    if (s->menu2D() && s->menu2D()->onMouseDown(button, x, y, mod))
        return true;

    // if menu is open close it
    if (SLButton::buttonParent && s->menu2D())
        s->menu2D()->closeAll();
   
    _mouseDownL = (button == ButtonLeft);
    _mouseDownR = (button == ButtonRight);
    _mouseDownM = (button == ButtonMiddle);
    _mouseMod = mod;
   
    SLbool result = false;
    result = _camera->onMouseDown(button, x, y, mod);
    for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
    {   if (s->_eventHandlers[i]->onMouseDown(button, x, y, mod))
            result = true;
    }  
    return result;
}  
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseUp gets called whenever a mouse button gets released.
*/
SLbool SLSceneView::onMouseUp(const SLMouseButton button, 
                              SLint x, SLint y, const SLKey mod)
{  
    SLScene* s = SLScene::current;
    _touchDowns = 0;
   
    if (_raytracer.state()==rtMoveGL)
    {   _renderType = renderRT;
        _raytracer.state(rtReady);
    }   
   
    // Check first if mouse up was on a button    
    if (s->menu2D() && s->menu2D()->onMouseUp(button, x, y, mod))
        return true;
           
    _mouseDownL = false;
    _mouseDownR = false;
    _mouseDownM = false;

    if (_camera && s->_root3D)
    {   SLbool result = false;
        result = _camera->onMouseUp(button, x, y, mod);
        for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
        {  if (s->_eventHandlers[i]->onMouseUp(button, x, y, mod))
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

    // save cursor position
    _posCursor.set(x, y);

    _touchDowns = 0;
    SLbool result = false;
      
    if (_mouseDownL || _mouseDownR || _mouseDownM)
    {   SLMouseButton btn = _mouseDownL ? ButtonLeft : 
                            _mouseDownR ? ButtonRight : ButtonMiddle;
      
        // Handle move in RT mode
        if (_renderType == renderRT && !_raytracer.continuous())
        {   if (_raytracer.state()==rtFinished)
                _raytracer.state(rtMoveGL);
            else
            {   _raytracer.continuous(false);
                s->menu2D(s->_menuGL);
                s->menu2D()->hideAndReleaseRec();
                s->menu2D()->drawBits()->off(SL_DB_HIDDEN);
            }
            _renderType = renderGL;
        }
      
        result = _camera->onMouseMove(btn, x, y, _mouseMod);

        for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
        {   if (s->_eventHandlers[i]->onMouseMove(btn, x, y, _mouseMod))
                result = true;
        }
    }  
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseWheel gets called whenever the mouse wheel is turned.
The parameter wheelPos is an increesing or decreesing counter number.
*/
SLbool SLSceneView::onMouseWheelPos(const SLint wheelPos, const SLKey mod)
{  
    static SLint lastMouseWheelPos = 0;
    SLint delta = wheelPos-lastMouseWheelPos;
    lastMouseWheelPos = wheelPos;
    return onMouseWheel(delta, mod);
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onMouseWheel gets called whenever the mouse wheel is turned.
The paramter delta is positive/negative depending on the wheel direction
*/
SLbool SLSceneView::onMouseWheel(const SLint delta, const SLKey mod)
{
    // Handle mousewheel in RT mode
    if (_renderType == renderRT && !_raytracer.continuous() && 
        _raytracer.state()==rtFinished)
        _raytracer.state(rtReady);

    SLScene* s = SLScene::current;
    SLbool result = false;

    // update active camera
    result = _camera->onMouseWheel(delta, mod);
    for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
    {   if (s->_eventHandlers[i]->onMouseWheel(delta, mod))
            result = true;
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onDoubleClick gets called when a mouse double click or finger 
double tab occures.
*/
SLbool SLSceneView::onDoubleClick(const SLMouseButton button, 
                                  SLint x, SLint y, const SLKey mod)
{  
    SLScene* s = SLScene::current;
   
    // Check first if mouse down was on a button    
    if (s->menu2D() && s->menu2D()->onDoubleClick(button, x, y, mod))
        return true;

    SLbool result = false;
   
    // Do object picking with ray cast
    if (button == ButtonLeft)
    {   _mouseDownR = false;
      
        SLRay pickRay;
        if (_camera) 
        {   _camera->eyeToPixelRay((SLfloat)x, (SLfloat)y, &pickRay);
            s->_root3D->hitRec(&pickRay);
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
        for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
        {   if (s->_eventHandlers[i]->onDoubleClick(button, x, y, mod))
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
SLbool SLSceneView::onTouch2Down(SLint x1, SLint y1, SLint x2, SLint y2)
{
    SLScene* s = SLScene::current;
    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
    _touchDowns = 2;
   
    SLbool result = false;
    result = _camera->onTouch2Down(x1, y1, x2, y2);
    for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
    {   if (s->_eventHandlers[i]->onTouch2Down(x1, y1, x2, y2))
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
    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
   
    SLbool result = false;
    if (_touchDowns==2)
    {  result = _camera->onTouch2Move(x1, y1, x2, y2);
        for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
        {  if (s->_eventHandlers[i]->onTouch2Move(x1, y1, x2, y2))
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
    _touch[0].set(x1, y1);
    _touch[1].set(x2, y2);
    _touchDowns = 0;
    SLbool result = false;
   
    result = _camera->onTouch2Up(x1, y1, x2, y2);
    for (SLuint i=0; i < s->_eventHandlers.size(); ++i)
    {   if (s->_eventHandlers[i]->onTouch2Up(x1, y1, x2, y2))
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
SLbool SLSceneView::onKeyPress(const SLKey key, const SLKey mod)
{  
    SLScene* s = SLScene::current;
    
    if (key=='N') return onCommand(cmdNormalsToggle);
    if (key=='P') return onCommand(cmdWireMeshToggle);
    if (key=='C') return onCommand(cmdFaceCullToggle);
    if (key=='T') return onCommand(cmdTextureToggle);
    if (key=='M') return onCommand(cmdMultiSampleToggle);
    if (key=='F') return onCommand(cmdFrustCullToggle);
    if (key=='B') return onCommand(cmdBBoxToggle);

    if (key==KeyEsc)
    {   if(_renderType == renderRT)
        {  _stopRT = true;
            return false;
        }
        else if(_renderType == renderPT)
        {  _stopPT = true;
            return false;
        }
        else return true; // end the program
    }

    SLbool result = false;
    if (key || mod)
    {   result = _camera->onKeyPress(key, mod);
        for (SLuint i=0; i < s->_eventHandlers.size(); ++i)
        {   if (s->_eventHandlers[i]->onKeyPress(key, mod))
            result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::onKeyRelease get called whenever a key is released.
*/
SLbool SLSceneView::onKeyRelease(const SLKey key, const SLKey mod)
{  
    SLScene* s = SLScene::current;
    SLbool result = false;
   
    if (key || mod)
    {   result = _camera->onKeyRelease(key, mod);
        for (SLuint i=0; i<s->_eventHandlers.size(); ++i)
        {  if (s->_eventHandlers[i]->onKeyRelease(key, mod))
                result = true;
        }
    }
    return result;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onCommand: Event handler for commands. Most key press or menu
commands are collected and dispatched here.
*/
SLbool SLSceneView::onCommand(const SLCmd cmd)
{
    SLScene* s = SLScene::current;
    SLNode* root3D = s->root3D();

    switch(cmd)
    {
        case cmdQuit:
            // @todo not a clean exit here. we need system access to stop the loop
            //        that isn't handled by us but by the window system. Would be solved
            //        if we had SLApplication::stop which stops the loop handled by
            //        SLApplication
            exit(0);
        case cmdAboutToggle:
            if (s->_menu2D)
            {   if (s->_menu2D == s->_menuGL)
                     s->_menu2D = s->_btnAbout;
                else s->_menu2D = s->_menuGL;
                return true;
            } else return false;
        case cmdHelpToggle:
            if (s->_menu2D)
            {   if (s->_menu2D == s->_menuGL)
                    s->_menu2D = s->_btnHelp;
                else s->_menu2D = s->_menuGL;
                return true;
            } else return false;
        case cmdCreditsToggle:
            if (s->_menu2D)
            {   if (s->_menu2D == s->_menuGL)
                    s->_menu2D = s->_btnCredits;
            else s->_menu2D = s->_menuGL;
            return true;
            } else return false;

        case cmdSceneSmallTest:
        case cmdSceneFigure:
        case cmdSceneLargeModel:
        case cmdSceneMeshLoad:
        case cmdSceneRevolver:
        case cmdSceneTextureFilter:
        case cmdSceneFrustumCull1:
        case cmdSceneFrustumCull2:
        case cmdSceneTextureBlend:
        case cmdScenePerVertexBlinn:
        case cmdScenePerPixelBlinn:
        case cmdScenePerVertexWave:
        case cmdSceneWater:
        case cmdSceneBumpNormal:  
        case cmdSceneBumpParallax:
        case cmdSceneEarth: 
        case cmdSceneMassAnimation:
        case cmdSceneRTSpheres:
        case cmdSceneRTMuttenzerBox:
        case cmdSceneRTSoftShadows:
        case cmdSceneRTDoF:        s->onLoad(this, (SLCmd)cmd); return false;

        case cmdUseSceneViewCamera:  switchToSceneViewCamera(); return true;
        case cmdStatsToggle:      _showStats = !_showStats; return true;
        case cmdSceneInfoToggle:  _showInfo = !_showInfo; return true;
        case cmdWaitEventsToggle: _waitEvents = !_waitEvents; return true;
        case cmdMultiSampleToggle:
            _doMultiSampling = !_doMultiSampling;
            _raytracer.aaSamples(_doMultiSampling?3:1);
            return true;
        case cmdFrustCullToggle:  _doFrustumCulling = !_doFrustumCulling; return true;
        case cmdDepthTestToggle:  _doDepthTest = !_doDepthTest; return true;

        case cmdNormalsToggle:   _drawBits.toggle(SL_DB_NORMALS);  return true;
        case cmdWireMeshToggle:  _drawBits.toggle(SL_DB_WIREMESH); return true;
        case cmdBBoxToggle:      _drawBits.toggle(SL_DB_BBOX);     return true;
        case cmdAxisToggle:      _drawBits.toggle(SL_DB_AXIS);     return true;
        case cmdVoxelsToggle:    _drawBits.toggle(SL_DB_VOXELS);   return true;
        case cmdFaceCullToggle:  _drawBits.toggle(SL_DB_CULLOFF);  return true;
        case cmdTextureToggle:   _drawBits.toggle(SL_DB_TEXOFF);   return true;
        case cmdAnimationToggle:
            _drawBits.toggle(SL_DB_ANIMOFF); 
            // if we toggle animations back on take the current time
            _lastFrameMS = s->timeMilliSec();
            return true;
      
        case cmdRenderOpenGL:
            _renderType = renderGL;
            s->menu2D(s->_menuGL);
            return true;
        case cmdRTContinuously:   
            _raytracer.continuous(!_raytracer.continuous());
            return true;
        case cmdRTDistributed:   
            _raytracer.distributed(!_raytracer.distributed());
            startRaytracing(5);
            return true;
        case cmdRT1: startRaytracing(1); return true;
        case cmdRT2: startRaytracing(2); return true;
        case cmdRT3: startRaytracing(3); return true;
        case cmdRT4: startRaytracing(4); return true;
        case cmdRT5: startRaytracing(5); return true;
        case cmdRT6: startRaytracing(6); return true;
        case cmdRT7: startRaytracing(7); return true;
        case cmdRT8: startRaytracing(8); return true;
        case cmdRT9: startRaytracing(9); return true;
        case cmdRT0: startRaytracing(0); return true;
        case cmdRTSaveImage: _raytracer.saveImage(); return true;

        case cmdPT1: startPathtracing(5, 1); return true;
        case cmdPT10: startPathtracing(5, 10); return true;
        case cmdPT50: startPathtracing(5, 50); return true;
        case cmdPT100: startPathtracing(5, 100); return true;
        case cmdPT500: startPathtracing(5, 500); return true;
        case cmdPT1000: startPathtracing(5, 1000); return true;
        case cmdPT5000: startPathtracing(5, 5000); return true;
        case cmdPT10000: startPathtracing(5, 100000); return true;
        case cmdPTSaveImage: _pathtracer.saveImage(); return true;

        default: break;
   }

    if (_camera)
    {           
        SLProjection prevProjection = _camera->projection();
        SLbool perspectiveChanged = prevProjection != (SLProjection)(cmd-cmdProjPersp);

        switch(cmd)
        {   case cmdProjPersp:
                _camera->projection(monoPerspective);
                if (_renderType == renderRT && !_raytracer.continuous() && 
                    _raytracer.state()==rtFinished)
                    _raytracer.state(rtReady);
                break;
            case cmdProjOrtho:
                _camera->projection(monoOrthographic);
                if (_renderType == renderRT && !_raytracer.continuous() && 
                    _raytracer.state()==rtFinished)
                    _raytracer.state(rtReady);
                break;
            case cmdProjSideBySide:    _camera->projection(stereoSideBySide); break;
            case cmdProjSideBySideP:   _camera->projection(stereoSideBySideP); break;
            case cmdProjSideBySideD:   _camera->projection(stereoSideBySideD); break;
            case cmdProjLineByLine:    _camera->projection(stereoLineByLine); break;
            case cmdProjColumnByColumn:_camera->projection(stereoColumnByColumn); break;
            case cmdProjPixelByPixel:  _camera->projection(stereoPixelByPixel); break;
            case cmdProjColorRC:       _camera->projection(stereoColorRC); break;
            case cmdProjColorRG:       _camera->projection(stereoColorRG); break;
            case cmdProjColorRB:       _camera->projection(stereoColorRB); break;
            case cmdProjColorYB:       _camera->projection(stereoColorYB); break;
      
            case cmdCamSpeedLimitInc:  _camera->speedLimit(_camera->speedLimit()*1.2f); return true;
            case cmdCamSpeedLimitDec:  _camera->speedLimit(_camera->speedLimit()*0.8f); return true;
            case cmdCamEyeSepInc:      _camera->onMouseWheel( 1, KeyCtrl); return true;
            case cmdCamEyeSepDec:      _camera->onMouseWheel(-1, KeyCtrl); return true;
            case cmdCamFocalDistInc:   _camera->onMouseWheel( 1, KeyShift); return true;
            case cmdCamFocalDistDec:   _camera->onMouseWheel(-1, KeyShift); return true;
            case cmdCamFOVInc:         _camera->onMouseWheel( 1, KeyAlt); return true;
            case cmdCamFOVDec:         _camera->onMouseWheel(-1, KeyAlt); return true;
            case cmdCamAnimTurnYUp:    _camera->camAnim(turntableYUp); return true;
            case cmdCamAnimTurnZUp:    _camera->camAnim(turntableZUp); return true;
            case cmdCamAnimWalkYUp:    _camera->camAnim(walkingYUp); return true;
            case cmdCamAnimWalkZUp:    _camera->camAnim(walkingZUp); return true;
            case cmdCamDeviceRotOn:    _camera->useDeviceRot(true); return true;
            case cmdCamDeviceRotOff:   _camera->useDeviceRot(false); return true;
            case cmdCamDeviceRotToggle:_camera->useDeviceRot(!_camera->useDeviceRot()); return true;
            case cmdCamReset:          _camera->resetToInitialState(); return true;
            default: break;
        }
                
        // special case code ???
        if (perspectiveChanged)
        {
            if (cmd == cmdProjSideBySideD) 
            {
                _vrMode = true;
                dpi(dpi()*2);
                SLButton::minMenuPos.set(_scrW*0.25f+100.0f, _scrH*0.5f-150.0f);
                rebuild2DMenus();
                if (onShowSysCursor)
                    onShowSysCursor(false);
            }
            else if (prevProjection == stereoSideBySideD)
            {
                _vrMode = false;
                dpi((SLint)((SLfloat)dpi()*0.5f));
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
/*!
SLSceneView::onRotation: Event handler for rotation change of a mobile
device with Euler angles for pitch, yaw and roll. 
With the parameter zeroYawAfterSec sets the time in seconds after
which the yaw angle is set to zero by subtracting the average yaw in this time.
*/
void SLSceneView::onRotationPYR(const SLfloat pitchRAD, 
                                const SLfloat yawRAD, 
                                const SLfloat rollRAD,
                                const SLfloat zeroYawAfterSec)
{
    //SL_LOG("onRotation: pitch: %3.1f, yaw: %3.1f, roll: %3.1f\n", 
    //       pitchRAD * SL_RAD2DEG, 
    //       yawRAD   * SL_RAD2DEG, 
    //       rollRAD  * SL_RAD2DEG);

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
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::onRotation: Event handler for rotation change of a mobile
device with rotation quaternion.
*/
void SLSceneView::onRotationQUAT(const SLfloat quatX, 
                                 const SLfloat quatY, 
                                 const SLfloat quatZ, 
                                 const SLfloat quatW)
{
    _deviceRotation.set(quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------


/*! 
SLSceneView::rebuild2DMenus force a rebuild of all 2d elements, might be needed
if dpi or other screenspace related parameters changed.
@todo the menu is still contained in the scene which partly breaks this behaviour
      for multiview applications.
*/
void SLSceneView::rebuild2DMenus()
{
    SLScene* s = SLScene::current;
    
    s->_menu2D = NULL;
    delete s->_menuGL;      s->_menuGL     = 0;
    delete s->_menuRT;      s->_menuRT     = 0;
    delete s->_menuPT;      s->_menuPT     = 0;
    delete s->_info;        s->_info       = 0;
    delete s->_infoGL;      s->_infoGL     = 0;
    delete s->_infoRT;      s->_infoRT     = 0;
    delete s->_btnAbout;    s->_btnAbout   = 0;
    delete s->_btnHelp;     s->_btnHelp    = 0;
    delete s->_btnCredits;  s->_btnCredits = 0;

    build2DMenus();
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
    if (s->_menu2D) return;

    // Get current camera projection
    SLProjection proj = _camera ? _camera->projection() : monoPerspective;
   
    // Get current camera animation
    SLCamAnim anim = _camera ? _camera->camAnim() : turntableYUp;

    // Get current camera device rotation usage
    SLbool useDeviceRot = _camera ? _camera->useDeviceRot() : true;

    // Set font size depending on DPI
    SLTexFont* f = SLTexFont::getFont(1.7f, _dpi);

    SLButton *mn1, *mn2, *mn3, *mn4, *mn5;    // submenu button pointer
    SLCmd curS = (SLCmd)s->_currentID;    // current scene number
   
    mn1 = new SLButton(this, ">", f, cmdMenu, false, false, 0, true, 0, 0, SLCol3f::COLBFH, 0.3f, centerCenter);
    mn1->drawBits()->off(SL_DB_HIDDEN);

    mn2 = new SLButton(this, "Load Scene >", f);
    mn1->addChild(mn2);
   
    mn3 = new SLButton(this, "General >", f);
    mn2->addChild(mn3);
    mn3->addChild(new SLButton(this, "SmallTest", f, cmdSceneSmallTest, true, curS==cmdSceneSmallTest, mn2));
    mn3->addChild(new SLButton(this, "Large Model", f, cmdSceneLargeModel, true, curS==cmdSceneLargeModel, mn2));
    mn3->addChild(new SLButton(this, "Figure", f, cmdSceneFigure, true, curS==cmdSceneFigure, mn2));
    mn3->addChild(new SLButton(this, "Mesh Loader", f, cmdSceneMeshLoad, true, curS==cmdSceneMeshLoad, mn2));
    mn3->addChild(new SLButton(this, "Texture Blending", f, cmdSceneTextureBlend, true, curS==cmdSceneTextureBlend, mn2));
    mn3->addChild(new SLButton(this, "Frustum Culling 1", f, cmdSceneFrustumCull1, true, curS==cmdSceneFrustumCull1, mn2));
    mn3->addChild(new SLButton(this, "Frustum Culling 2", f, cmdSceneFrustumCull2, true, curS==cmdSceneFrustumCull2, mn2));
    mn3->addChild(new SLButton(this, "Texture Filtering", f, cmdSceneTextureFilter, true, curS==cmdSceneTextureFilter, mn2));
    mn3->addChild(new SLButton(this, "Mass Animation", f, cmdSceneMassAnimation, true, curS==cmdSceneMassAnimation, mn2));

    mn3 = new SLButton(this, "Shader >", f);
    mn2->addChild(mn3);
    mn3->addChild(new SLButton(this, "Per Vertex Lighting", f, cmdScenePerVertexBlinn, true, curS==cmdScenePerVertexBlinn, mn2));
    mn3->addChild(new SLButton(this, "Per Pixel Lighting", f, cmdScenePerPixelBlinn, true, curS==cmdScenePerPixelBlinn, mn2));
    mn3->addChild(new SLButton(this, "Per Vertex Wave", f, cmdScenePerVertexWave, true, curS==cmdScenePerVertexWave, mn2));
    mn3->addChild(new SLButton(this, "Water", f, cmdSceneWater, true, curS==cmdSceneWater, mn2));
    mn3->addChild(new SLButton(this, "Bump Mapping", f, cmdSceneBumpNormal, true, curS==cmdSceneBumpNormal, mn2, true));
    mn3->addChild(new SLButton(this, "Parallax Mapping", f, cmdSceneBumpParallax, true, curS==cmdSceneBumpParallax, mn2));
    mn3->addChild(new SLButton(this, "Glass Shader", f, cmdSceneRevolver, true, curS==cmdSceneRevolver, mn2));
    mn3->addChild(new SLButton(this, "Earth Shader", f, cmdSceneEarth, true, curS==cmdSceneEarth, mn2));
   
    mn3 = new SLButton(this, "Ray tracing >", f);
    mn2->addChild(mn3);
    mn3->addChild(new SLButton(this, "Spheres", f, cmdSceneRTSpheres, true, curS==cmdSceneRTSpheres, mn2));
    mn3->addChild(new SLButton(this, "Muttenzer Box", f, cmdSceneRTMuttenzerBox, true, curS==cmdSceneRTMuttenzerBox, mn2));
    mn3->addChild(new SLButton(this, "Soft Shadows", f, cmdSceneRTSoftShadows, true, curS==cmdSceneRTSoftShadows, mn2));
    mn3->addChild(new SLButton(this, "Depth of Field", f, cmdSceneRTDoF, true, curS==cmdSceneRTDoF, mn2));

    mn2 = new SLButton(this, "Camera >", f); mn1->addChild(mn2);
   
    mn2->addChild(new SLButton(this, "Reset", f, cmdCamReset));
    
    mn3 = new SLButton(this, "Projection >", f);
    mn2->addChild(mn3);
    for (SLint p=monoPerspective; p<=monoOrthographic; ++p)
    {   mn3->addChild(new SLButton(this, SLCamera::projectionToStr((SLProjection)p), f, 
                                   (SLCmd)(cmdProjPersp+p), true, proj==p, mn3));
    }

    mn4 = new SLButton(this, "Stereo >", f);
    mn3->addChild(mn4);

    mn5 = new SLButton(this, "Eye separation >", f);
    mn4->addChild(mn5);
    mn5->addChild(new SLButton(this, "-10%", f, cmdCamEyeSepDec, false, false, 0, false));
    mn5->addChild(new SLButton(this, "+10%", f, cmdCamEyeSepInc, false, false, 0, false));

    mn5 = new SLButton(this, "Focal dist. >", f);
    mn4->addChild(mn5);
    mn5->addChild(new SLButton(this, "+5%", f, cmdCamFocalDistInc, false, false, 0, false));
    mn5->addChild(new SLButton(this, "-5%", f, cmdCamFocalDistDec, false, false, 0, false));

    for (SLint p=stereoSideBySide; p<=stereoColorYB; ++p)
    {   mn4->addChild(new SLButton(this, SLCamera::projectionToStr((SLProjection)p), f, 
                                   (SLCmd)(cmdProjPersp+p), true, proj==p, mn3));
    }
   
    mn3 = new SLButton(this, "Animation >", f);
    mn2->addChild(mn3);
    mn3->addChild(new SLButton(this, "Walking Y up",   f, cmdCamAnimWalkYUp, true, anim==walkingYUp, mn3));
    mn3->addChild(new SLButton(this, "Walking Z up",   f, cmdCamAnimWalkZUp, true, anim==walkingYUp, mn3));
    mn3->addChild(new SLButton(this, "Turntable Y up", f, cmdCamAnimTurnYUp, true, anim==turntableYUp, mn3));
    mn3->addChild(new SLButton(this, "Turntable Z up", f, cmdCamAnimTurnZUp, true, anim==turntableZUp, mn3));
      
    mn3 = new SLButton(this, "View Angle >", f);
    mn2->addChild(mn3);
    mn3->addChild(new SLButton(this, "+5 deg.", f, cmdCamFOVInc, false, false, 0, false));
    mn3->addChild(new SLButton(this, "-5 deg.", f, cmdCamFOVDec, false, false, 0, false));
   
    mn3 = new SLButton(this, "Walk Speed >", f);
    mn2->addChild(mn3);
    mn3->addChild(new SLButton(this, "+20%", f, cmdCamSpeedLimitInc, false, false, 0, false));
    mn3->addChild(new SLButton(this, "-20%", f, cmdCamSpeedLimitDec, false, false, 0, false));

    mn3 = new SLButton(this, "Use Device Rotation", f, cmdCamDeviceRotToggle, true, useDeviceRot, 0, false);
    mn2->addChild(mn3);
   
    mn2 = new SLButton(this, "Render Flags >", f);
    mn1->addChild(mn2);
    mn2->addChild(new SLButton(this, "Slowdown on Idle", f, cmdWaitEventsToggle, true, _waitEvents, 0, false));
    if (hasMultiSampling())
        mn2->addChild(new SLButton(this, "Do Multisampling", f, cmdMultiSampleToggle, true, _doMultiSampling, 0, false));
    mn2->addChild(new SLButton(this, "Do Frustum Culling", f, cmdFrustCullToggle, true, _doFrustumCulling, 0, false));
    mn2->addChild(new SLButton(this, "Do Depth Test", f, cmdDepthTestToggle, true, _doDepthTest, 0, false));
    mn2->addChild(new SLButton(this, "Animation off", f, cmdAnimationToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "Textures off", f, cmdTextureToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "Back faces", f, cmdFaceCullToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "AABB", f, cmdBBoxToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "Axis", f, cmdAxisToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "Voxels", f, cmdVoxelsToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "Normals", f, cmdNormalsToggle, true, false, 0, false));
    mn2->addChild(new SLButton(this, "Wiremesh", f, cmdWireMeshToggle, true, false, 0, false));
   
    mn2 = new SLButton(this, "Infos >", f);
    mn1->addChild(mn2);
    mn2->addChild(new SLButton(this, "About", f, cmdAboutToggle));
    mn2->addChild(new SLButton(this, "Help", f, cmdHelpToggle));
    mn2->addChild(new SLButton(this, "Credits", f, cmdCreditsToggle));
    mn2->addChild(new SLButton(this, "Scene Info", f, cmdSceneInfoToggle, true, _showInfo));
    mn2->addChild(new SLButton(this, "Statistics", f, cmdStatsToggle, true, _showStats));

    mn2 = new SLButton(this, "Renderer >", f);
    mn1->addChild(mn2);
    mn2->addChild(new SLButton(this, "Ray tracing", f, cmdRT5, false, false, 0, true));
    #ifndef SL_GLES2
    mn2->addChild(new SLButton(this, "Path tracing", f, cmdPT10, false, false, 0, true));
    #else
    mn2->addChild(new SLButton(this, "Path tracing", f, cmdPT1, false, false, 0, true));
    #endif

    mn2 = new SLButton(this, "Quit", f, cmdQuit);
    mn1->addChild(mn2);

    // Init OpenGL menu
    _stateGL->modelViewMatrix.identity();
    mn1->setSizeRec();
    mn1->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    mn1->updateAABBRec();
    s->_menuGL = mn1;
   
    // Build ray tracing menu
    SLCol3f green(0.0f,0.5f,0.0f);
   
    mn1 = new SLButton(this, ">", f, cmdMenu, false, false, 0, true,  0, 0, green, 0.3f, centerCenter);
    mn1->drawBits()->off(SL_DB_HIDDEN);
   
    mn1->addChild(new SLButton(this, "OpenGL Rendering", f, cmdRenderOpenGL, false, false, 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Render continuously", f, cmdRTContinuously, true, _raytracer.continuous(), 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Render parallel distributed", f, cmdRTDistributed, true, _raytracer.distributed(), 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Rendering Depth 1", f, cmdRT1, false, false, 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Rendering Depth 5", f, cmdRT5, false, false, 0, true,  0, 0, green));
    mn1->addChild(new SLButton(this, "Rendering Depth max.", f, cmdRT0, false, false, 0, true,  0, 0, green));
    #ifndef SL_GLES2
    mn1->addChild(new SLButton(this, "Save Image", f, cmdRTSaveImage, false, false, 0, true,  0, 0, green));
    #endif

    // Init RT menu
    _stateGL->modelViewMatrix.identity();
    mn1->setSizeRec();
    mn1->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    mn1->updateAABBRec();
    s->_menuRT = mn1;

    // Build path tracing menu
    SLCol3f blue(0.0f,0.0f,0.5f);
   
    mn1 = new SLButton(this, ">", f, cmdMenu, false, false, 0, true,  0, 0, blue, 0.3f, centerCenter);
    mn1->drawBits()->off(SL_DB_HIDDEN);

    mn1->addChild(new SLButton(this, "OpenGL Rendering", f, cmdRenderOpenGL, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "1 Sample Ray", f, cmdPT1, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "10 Sample Rays", f, cmdPT10, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "50 Sample Rays", f, cmdPT50, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "100 Sample Rays", f, cmdPT100, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "500 Sample Rays", f, cmdPT500, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "1000 Sample Rays", f, cmdPT1000, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "5000 Sample Rays", f, cmdPT5000, false, false, 0, true,  0, 0, blue));
    mn1->addChild(new SLButton(this, "10000 Sample Rays", f, cmdPT10000, false, false, 0, true,  0, 0, blue));
    #ifndef SL_GLES2
    mn1->addChild(new SLButton(this, "Save Image", f, cmdPTSaveImage, false, false, 0, true,  0, 0, blue));
    #endif

    // Init PT menu
    _stateGL->modelViewMatrix.identity();
    mn1->setSizeRec();
    mn1->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    mn1->updateAABBRec();
    s->_menuPT = mn1;

    build2DMsgBoxes(); 

    // if menu is initially visible show first the about button
    if (_showMenu)
         s->_menu2D = s->_btnAbout;
    else s->_menu2D = s->_menuGL;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::build2DInfoGL builds the _infoGL groups with info texts. 
See SLButton and SLText class for more infos. 
*/
void SLSceneView::build2DInfoGL()
{
    SLScene* s = SLScene::current;
    if (s->_infoGL) delete s->_infoGL;
   
    // prepare some statistic infos
    SLCamera* cam = camera();
    SLfloat vox = (SLfloat)_stats.numVoxels;
    SLfloat voxEmpty = (SLfloat)_stats.numVoxEmpty;
    SLfloat voxelsEmpty  = vox ? voxEmpty / vox*100.0f : 0.0f;
    SLfloat numRTTria = (SLfloat)_stats.numTriangles;
    SLfloat avgTriPerVox = vox ? numRTTria / (vox-voxEmpty) : 0.0f;
    SLfloat updateTimePC = _updateTimeMS.average()/_frameTimeMS.average()*100.0f;
    SLfloat cullTimePC = _cullTimeMS.average()/_frameTimeMS.average()*100.0f;
    SLfloat draw3DTimePC = _draw3DTimeMS.average()/_frameTimeMS.average()*100.0f;
    SLfloat draw2DTimePC = _draw2DTimeMS.average()/_frameTimeMS.average()*100.0f;
    SLfloat eyeSepPC = cam->eyeSeparation()/cam->focalDist()*100;
   
    SLchar m[2550];   // message charcter array
    m[0]=0;           // set zero length
    sprintf(m+strlen(m), "Scene: %s\\n", s->name().c_str());
    sprintf(m+strlen(m), "DPI: %d\\n", _dpi);
    sprintf(m+strlen(m), "FPS: %4.1f  (Size: %d x %d)\\n", _fps, _scrW, _scrH);
    sprintf(m+strlen(m), "Frame Time : %4.1f ms\\n", _frameTimeMS.average());
    sprintf(m+strlen(m), "Update Time : %4.1f ms (%0.0f%%)\\n", _updateTimeMS.average(), updateTimePC);
    sprintf(m+strlen(m), "Culling Time : %4.1f ms (%0.0f%%)\\n", _cullTimeMS.average(), cullTimePC);
    sprintf(m+strlen(m), "Draw Time 3D: %4.1f ms (%0.0f%%)\\n", _draw3DTimeMS.average(), draw3DTimePC);
    sprintf(m+strlen(m), "Draw Time 2D: %4.1f ms (%0.0f%%)\\n", _draw2DTimeMS.average(), draw2DTimePC);
    sprintf(m+strlen(m), "Shapes in Frustum: %d\\n", cam->numRendered());
    sprintf(m+strlen(m), "NO. of drawcalls: %d\\n", _totalDrawCalls);
    sprintf(m+strlen(m), "--------------------------------------------\\n");
    sprintf(m+strlen(m), "OpenGL: %s\\n", _stateGL->glVersion().c_str());
    sprintf(m+strlen(m), "Vendor: %s\\n", _stateGL->glVendor().c_str());
    sprintf(m+strlen(m), "Renderer: %s\\n", _stateGL->glRenderer().c_str());
    sprintf(m+strlen(m), "GLSL: %s\\n", _stateGL->glGLSLVersion().c_str());
    sprintf(m+strlen(m), "--------------------------------------------\\n");
    sprintf(m+strlen(m), "Projection: %s\\n", cam->projectionStr().c_str());
    sprintf(m+strlen(m), "Animation: %s\\n", cam->animationStr().c_str());
    sprintf(m+strlen(m), "Speed Limit: %4.1f/sec.\\n", cam->speedLimit());
    if (camera()->projection() > monoOrthographic)
        sprintf(m+strlen(m), "Eye separation: %4.2f (%3.1f%% of f)\\n", cam->eyeSeparation(), eyeSepPC);
    sprintf(m+strlen(m), "fov: %4.2f\\n", cam->fov());
    sprintf(m+strlen(m), "Focal distance (f): %4.2f\\n", cam->focalDist());
    sprintf(m+strlen(m), "Projection size: %4.2f x %4.2f\\n", cam->focalDistScrW(), cam->focalDistScrH());
    sprintf(m+strlen(m), "--------------------------------------------\\n");
    sprintf(m+strlen(m), "No. of Group/Leaf Nodes: %d / %d\\n", _stats.numGroupNodes,  _stats.numLeafNodes);
    sprintf(m+strlen(m), "Lights: %d\\n", _stats.numLights);
    sprintf(m+strlen(m), "MB in Meshes: %4.2f\\n", (SLfloat)_stats.numBytes / 1000000.0f);
    sprintf(m+strlen(m), "MB in Accel.: %4.2f\\n", (SLfloat)_stats.numBytesAccel / 1000000.0f);
    sprintf(m+strlen(m), "No. of VBOs/MB: %u / %4.2f\\n", _totalBufferCount, (SLfloat)_totalBufferSize / 1000000.0f);
    sprintf(m+strlen(m), "No. of Voxels/empty: %d / %4.1f%%\\n", _stats.numVoxels, voxelsEmpty);
    sprintf(m+strlen(m), "Avg. Tria/Voxel: %4.1f\\n", avgTriPerVox);
    sprintf(m+strlen(m), "Max. Tria/Voxel: %d\\n", _stats.numVoxMaxTria);
    sprintf(m+strlen(m), "Group Nodes: %u\\n", _stats.numGroupNodes);
    sprintf(m+strlen(m), "Leaf Nodes: %u\\n", _stats.numLeafNodes);
    sprintf(m+strlen(m), "Meshes: %u\\n", _stats.numMeshes);
    sprintf(m+strlen(m), "Triangles: %u\\n", _stats.numTriangles);

    SLTexFont* f = SLTexFont::getFont(1.2f, _dpi);
    SLText* t = new SLText(m, f, SLCol4f::WHITE, (SLfloat)_scrW, 1.0f);
    t->translate(10.0f, -t->size().y-5.0f, 0.0f, TS_Local);
    s->_infoGL = t;
}
//-----------------------------------------------------------------------------
/*! 
SLSceneView::build2DInfoRT builds the _infoRT group with info texts. 
See SLButton and SLText class for more infos. 
*/
void SLSceneView::build2DInfoRT()
{     
    SLScene* s = SLScene::current;
    if (s->_infoRT) 
        delete s->_infoRT;
   
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
   
    SLchar m[2550];   // message charcter array
    m[0]=0;           // set zero length
    sprintf(m+strlen(m), "Scene: %s\\n", s->name().c_str());
    sprintf(m+strlen(m), "Time per frame: %4.2f sec.  (Size: %d x %d)\\n", rt->renderSec(), _scrW, _scrH);
    sprintf(m+strlen(m), "fov: %4.2f\\n", cam->fov());
    sprintf(m+strlen(m), "Focal dist. (f): %4.2f\\n", cam->focalDist());
    sprintf(m+strlen(m), "--------------------------------------------\\n");
    sprintf(m+strlen(m), "Threads: %d\\n", rt->numThreads());
    sprintf(m+strlen(m), "Max. allowed RT depth: %d\\n", SLRay::maxDepth);
    sprintf(m+strlen(m), "Max. reached RT depth: %d\\n", SLRay::maxDepthReached);
    sprintf(m+strlen(m), "Average RT depth: %4.2f\\n", SLRay::avgDepth/primaries);
    sprintf(m+strlen(m), "AA threshhold: %2.1f\\n", rt->aaThreshold());
    sprintf(m+strlen(m), "AA samples: %d x %d\\n", rt->aaSamples(), rt->aaSamples());
    sprintf(m+strlen(m), "AA pixels: %u, %3.1f%%\\n", SLRay::subsampledPixels, (SLfloat)SLRay::subsampledPixels/primaries*100.0f);
    sprintf(m+strlen(m), "Primary rays: %u, %3.1f%%\\n", primaries, (SLfloat)primaries/total*100.0f);
    sprintf(m+strlen(m), "Reflected rays: %u, %3.1f%%\\n", SLRay::reflectedRays, (SLfloat)SLRay::reflectedRays/total*100.0f);
    sprintf(m+strlen(m), "Refracted rays: %u, %3.1f%%\\n", SLRay::refractedRays, (SLfloat)SLRay::refractedRays/total*100.0f);
    sprintf(m+strlen(m), "TIR rays: %u, %3.1f%%\\n", SLRay::tirRays, (SLfloat)SLRay::tirRays/total*100.0f);
    sprintf(m+strlen(m), "Shadow rays: %u, %3.1f%%\\n", SLRay::shadowRays, (SLfloat)SLRay::shadowRays/total*100.0f);
    sprintf(m+strlen(m), "AA rays: %u, %3.1f%%\\n", SLRay::subsampledRays, (SLfloat)SLRay::subsampledRays/total*100.0f);
    sprintf(m+strlen(m), "Total rays: %u, %3.1f%%\\n", total, 100.0f);
    sprintf(m+strlen(m), "Intersection tests: %u\\n", SLRay::intersections);
    sprintf(m+strlen(m), "Intersections: %u, %3.1f%%\\n", SLRay::tests, SLRay::intersections/(SLfloat)SLRay::tests*100.0f);
    sprintf(m+strlen(m), "--------------------------------------------\\n");
    sprintf(m+strlen(m), "Group Nodes: %d\\n", _stats.numGroupNodes);
    sprintf(m+strlen(m), "Leaf Nodes: %d\\n", _stats.numLeafNodes);
    sprintf(m+strlen(m), "Lights: %d\\n", _stats.numLights);
    sprintf(m+strlen(m), "MB in Meshes: %f\\n", (SLfloat)_stats.numBytes / 1000000.0f);
    sprintf(m+strlen(m), "MB in Accel.: %f\\n", (SLfloat)_stats.numBytesAccel / 1000000.0f);
    sprintf(m+strlen(m), "Triangles: %d\\n", _stats.numTriangles);
    sprintf(m+strlen(m), "Voxels: %d\\n", _stats.numVoxels);
    sprintf(m+strlen(m), "Voxels empty: %4.1f%%\\n", voxelsEmpty);
    sprintf(m+strlen(m), "Avg. Tria/Voxel: %4.1f\\n", avgTriPerVox);
    sprintf(m+strlen(m), "Max. Tria/Voxel: %d", _stats.numVoxMaxTria);
   
    SLTexFont* f = SLTexFont::getFont(1.2f, _dpi);
    SLText* t = new SLText(m, f, SLCol4f::WHITE, (SLfloat)_scrW, 1.0f);
    t->translate(10.0f, -t->size().y-5.0f, 0.0f, TS_Local);
    s->_infoRT = t;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::build2DInfoLoading builds the _infoLoading info texts.
See SLButton and SLText class for more infos.
*/
void SLSceneView::build2DInfoLoading()
{
    SLScene* s = SLScene::current;
    if (s->_infoLoading) return;
    SLTexFont* f = SLTexFont::getFont(3, _dpi);
    SLText* t = new SLText("Loading Scene . . .", f, SLCol4f::WHITE, (SLfloat)_scrW, 1.0f);
    t->translate(10.0f, -t->size().y-5.0f, 0.0f, TS_Local);
    t->translate(_scrW*0.5f - t->size().x*0.5f, -(_scrH*0.5f) + t->size().y, 0.0f, TS_Local);
    s->_infoLoading = t;
}
//-----------------------------------------------------------------------------
/*!
Builds the message buttons. They depend on screen width.
*/
void SLSceneView::build2DMsgBoxes()
{ 
    SLScene*    s = SLScene::current;
    SLTexFont*  f = SLTexFont::getFont(1.7f, _dpi);
   
    // Help button
    if (s->_btnHelp) delete s->_btnHelp;
    s->_btnHelp = new SLButton(this, s->_infoHelp_en, f,
                               cmdAboutToggle, false, false, 0, true,
                               _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                               SLCol3f::COLBFH, 0.8f, centerCenter);

    _stateGL->modelViewMatrix.identity();
    s->_btnHelp->drawBits()->off(SL_DB_HIDDEN);
    s->_btnHelp->setSizeRec();
    s->_btnHelp->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->_btnHelp->updateAABBRec();
   
    // About button
    if (s->_btnAbout) delete s->_btnAbout;
    s->_btnAbout = new SLButton(this, s->_infoAbout_en, f,
                                cmdAboutToggle, false, false, 0, true,
                                _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                                SLCol3f::COLBFH, 0.8f, centerCenter);

    _stateGL->modelViewMatrix.identity();
    s->_btnAbout->drawBits()->off(SL_DB_HIDDEN);
    s->_btnAbout->setSizeRec();
    s->_btnAbout->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->_btnAbout->updateAABBRec();
   
    // Credits button
    if (s->_btnCredits) delete s->_btnCredits;
    s->_btnCredits = new SLButton(this, s->_infoCredits_en, f,
                                    cmdAboutToggle, false, false, 0, true,
                                    _scrW - 2*SLButton::minMenuPos.x, 0.0f,
                                    SLCol3f::COLBFH, 0.8f, centerCenter);

    _stateGL->modelViewMatrix.identity();
    s->_btnCredits->drawBits()->off(SL_DB_HIDDEN);
    s->_btnCredits->setSizeRec();
    s->_btnCredits->setPosRec(SLButton::minMenuPos.x, SLButton::minMenuPos.y);
    s->_btnCredits->updateAABBRec();
}
//-----------------------------------------------------------------------------




/*! 
SLSceneView::calcFPS calculates the precise frame per second rate. 
The calculation is done every half second and averaged with the last FPS calc.
*/
SLfloat SLSceneView::calcFPS(SLfloat deltaTimeMS)
{
    _frameTimeMS.set(deltaTimeMS);
    _fps = 1 / _frameTimeMS.average() * 1000.0f;
    if (_fps < 0.0f) _fps = 0.0f;
    return _fps;
}
//-----------------------------------------------------------------------------
/*!
Returns the window title with name & FPS
*/
SLstring SLSceneView::windowTitle()
{  
    SLScene* s = SLScene::current;
    SLchar title[255];
    if (!_camera || !s->_root3D)
        return SLstring("-");

    if (_renderType == renderRT)
    {   if (_raytracer.continuous())
        {   sprintf(title, "%s (fps: %4.1f, Threads: %d)", 
                    s->name().c_str(), 
                    _fps, 
                    _raytracer.numThreads());
        } else
        {   sprintf(title, "%s (%d%%, Threads: %d)", 
                    s->name().c_str(), 
                    _raytracer.pcRendered(), 
                    _raytracer.numThreads());
        }
    } else
    if (_renderType == renderPT)
    {   sprintf(title, "%s (%d%%, Threads: %d)", 
                s->name().c_str(), 
                _pathtracer.pcRendered(), 
                _pathtracer.numThreads());
    } else
    {   sprintf(title, "%s (fps: %4.1f, %u shapes rendered)", 
                        s->name().c_str(), 
                        _fps,
                        _camera->numRendered());
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
    _renderType = renderRT;
    _stopRT = false;
    _raytracer.maxDepth(maxDepth);
    _raytracer.aaSamples(_doMultiSampling?3:1);
    s->_menu2D = s->_menuRT;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::updateAndRT3D starts the raytracing or refreshes the current RT
image during rendering. The function returns true if an animation was done 
prior to the rendering start.
*/
SLbool SLSceneView::updateAndDrawRT3D(SLfloat elapsedTimeMS)
{
    SLScene* s = SLScene::current;
    SLbool updated = false;
   
    // if the raytracer not yet got started
    if (_raytracer.state()==rtReady)
    {
        if (s->_root3D)
        {   updated = !drawBit(SL_DB_ANIMOFF) && 
                      !s->_root3D->drawBit(SL_DB_ANIMOFF) && 
                       s->_root3D->animateRec(40);
        }
                
        _stateGL->modelViewMatrix.identity();
        s->_root3D->updateAABBRec();

        // Start raytracing
        if (_raytracer.distributed())
             _raytracer.renderDistrib(this);
        else _raytracer.renderClassic(this);
    }

    // Refresh the render image during RT
    _raytracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if(_stopRT)
    {   _renderType = renderGL;
        s->menu2D(s->_menuGL);
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
    _renderType = renderPT;
    _stopPT = false;
    _pathtracer.maxDepth(maxDepth);
    _pathtracer.aaSamples(samples);
    s->_menu2D = s->_menuPT;
}
//-----------------------------------------------------------------------------
/*!
SLSceneView::updateAndRT3D starts the raytracing or refreshes the current RT
image during rendering. The function returns true if an animation was done 
prior to the rendering start.
*/
SLbool SLSceneView::updateAndDrawPT3D(SLfloat elapsedTimeMS)
{
    SLScene* s = SLScene::current;
    SLbool updated = false;
   
    // if the pathtracer not yet got started
    if (_pathtracer.state()==rtReady)
    {
        updated = !drawBit(SL_DB_ANIMOFF) && 
                      !s->_root3D->drawBit(SL_DB_ANIMOFF) && 
                       s->_root3D->animateRec(40);

        
        _stateGL->modelViewMatrix.identity();
        s->_root3D->updateAABBRec();

        // Start raytracing
        _pathtracer.render(this);
    }

    // Refresh the render image during PT
    _pathtracer.renderImage();

    // React on the stop flag (e.g. ESC)
    if(_stopPT)
    {   _renderType = renderGL;
        updated = true;
    }

    return updated;
}
//-----------------------------------------------------------------------------
//! Sets the loading text flags
void SLSceneView::showLoading(SLbool showLoading) 
{
    if (showLoading)
    {   _stateGL->clearColor(SLCol4f::GRAY);
        _stateGL->clearColorDepthBuffer();
    }
    _showLoading = showLoading;
}
//------------------------------------------------------------------------------
