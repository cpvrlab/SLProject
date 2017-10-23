//#############################################################################
//  File:      SLScene.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#ifdef SL_MEMLEAKDETECT     // set in SL.h for debug config only
#include <debug_new.h>      // memory leak detector
#endif

#include <SLScene.h>
#include <SLSceneView.h>
#include <SLText.h>
#include <SLInputManager.h>
#include <SLCVCapture.h>
#include <SLAssimpImporter.h>
#include <SLLightDirect.h>
#include <SLCVTracked.h>
#include <SLCVTrackedAruco.h>

//-----------------------------------------------------------------------------
/*! Global static scene pointer that can be used throughout the entire library
to access the current scene and its sceneviews. 
*/
SLScene* SLScene::current = nullptr;
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
  - app-Demo-GLFW: glfwMain.cpp in function main()
  - app-Demo-Qt: qtGLWidget::initializeGL()
  - app-Viewer-Qt: qtGLWidget::initializeGL()
  - app-Demo-Android: Java_ch_fhnw_comgRT_glES2Lib_onInit()
  - app-Demo-iOS: ViewController.m in method viewDidLoad()
*/
SLScene::SLScene(SLstring name) : SLObject(name)
{  
    current = this;

    _root3D         = nullptr;
    _root2D         = nullptr;
    _info           = "";
    _selectedMesh   = nullptr;
    _selectedNode   = nullptr;
    _activeCalib    = nullptr;
    _stopAnimations = false;
    _usesRotation   = false;
    _fps            = 0;
    _elapsedTimeMS  = 0;
    _lastUpdateTimeMS = 0;
     
    // Load std. shader programs in order as defined in SLShaderProgs enum in SLenum
    // In the constructor they are added the _shaderProgs vector
    // If you add a new shader here you have to update the SLShaderProgs enum accordingly.
    SLGLProgram* p;
    p = new SLGLGenericProgram("ColorAttribute.vert","Color.frag");
    p = new SLGLGenericProgram("ColorUniform.vert","Color.frag");
    p = new SLGLGenericProgram("PerVrtBlinn.vert","PerVrtBlinn.frag");
    p = new SLGLGenericProgram("PerVrtBlinnColorAttrib.vert","PerVrtBlinn.frag");
    p = new SLGLGenericProgram("PerVrtBlinnTex.vert","PerVrtBlinnTex.frag");
    p = new SLGLGenericProgram("TextureOnly.vert","TextureOnly.frag");
    p = new SLGLGenericProgram("PerPixBlinn.vert","PerPixBlinn.frag");
    p = new SLGLGenericProgram("PerPixBlinnTex.vert","PerPixBlinnTex.frag");
    p = new SLGLGenericProgram("PerPixCookTorrance.vert","PerPixCookTorrance.frag");
    p = new SLGLGenericProgram("PerPixCookTorranceTex.vert","PerPixCookTorranceTex.frag");
    p = new SLGLGenericProgram("BumpNormal.vert","BumpNormal.frag");
    p = new SLGLGenericProgram("BumpNormal.vert","BumpNormalParallax.frag");
    p = new SLGLGenericProgram("FontTex.vert","FontTex.frag");
    p = new SLGLGenericProgram("StereoOculus.vert","StereoOculus.frag");
    p = new SLGLGenericProgram("StereoOculusDistortionMesh.vert","StereoOculusDistortionMesh.frag");
    _numProgsPreload = (SLint)_programs.size();
   
    // font and video texture are not added to the _textures vector
    SLTexFont::generateFonts();

    // load default video image that is displayed when no live video is available
    _videoTexture.setVideoImage("LiveVideoError.png");

    // Set video type to none (this also sets the active calibration to the main calibration)
    videoType(VT_NONE);

    // load opencv camera calibration for main and secondary camera
    #if defined(SL_USES_CVCAPTURE)
    _calibMainCam.load("cam_calibration_main.xml", true, false);
    _calibMainCam.loadCalibParams();
    _activeCalib = &_calibMainCam;
    SLCVCapture::hasSecondaryCamera = false;
    #else
    _calibMainCam.load("cam_calibration_main.xml", false, false);
    _calibMainCam.loadCalibParams();
    _calibScndCam.load("cam_calibration_scnd.xml", true, false);
    _calibScndCam.loadCalibParams();
    _activeCalib = &_calibMainCam;
    SLCVCapture::hasSecondaryCamera = true;
    #endif

    _showDetection = false;

    _oculus.init();
}
//-----------------------------------------------------------------------------
/*! The destructor does the final total deallocation of all global resources.
The destructor is called in slTerminate.
*/
SLScene::~SLScene()
{
    // load opencv camera calibration for main and secondary camera
    #if defined(SL_USES_CVCAPTURE)
    _calibMainCam.save();
    #else
    _calibMainCam.save();
    _calibScndCam.save();
    #endif

    // Delete all remaining sceneviews
    for (auto sv : _sceneViews)
        if (sv != nullptr)
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
        
    // delete AR tracker programs
    for (auto t : _trackers) delete t;
    _trackers.clear();
   
    // delete fonts   
    SLTexFont::deleteFonts();
   
    current = nullptr;

    #ifdef SL_USES_CVCAPTURE
    // release the capture device
    SLCVCapture::release();
    #endif

    SL_LOG("Destructor      : ~SLScene\n");
    SL_LOG("------------------------------------------------------------------\n");
}
//-----------------------------------------------------------------------------
/*! The scene init is called whenever the scene is new loaded.
*/
void SLScene::init()
{     
    unInit();
   
    _globalAmbiLight.set(0.2f,0.2f,0.2f,0.0f);
    _selectedNode = 0;

    _timer.start();
    _frameTimesMS.init();
    _updateTimesMS.init();
    _cullTimesMS.init();
    _draw3DTimesMS.init();
    _draw2DTimesMS.init();
    _trackingTimesMS.init();
    _detectTimesMS.init();
    _matchTimesMS.init();
    _optFlowTimesMS.init();
    _poseTimesMS.init();
    _captureTimesMS.init(200);

//todo    _deviceRotation  = SLQuat4f::IDENTITY;
    _deviceRotation.identity();
    _devicePitchRAD  = 0.0f;
    _deviceYawRAD    = 0.0f;
    _deviceRollRAD   = 0.0f;
    _zeroYawAtStart  = true;
    _startYawRAD     = 0.0f;
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
        if (sv != nullptr)
            sv->camera(sv->sceneViewCamera());

    // delete entire scene graph
    delete _root3D;
    _root3D = nullptr;
    delete _root2D;
    _root2D = nullptr;

    // clear light pointers
    _lights.clear();

    // delete textures
    for (auto t : _textures) delete t;
    _textures.clear();
   
    // manually clear the default materials (it will get deleted below)
    SLMaterial::defaultGray(nullptr);
    SLMaterial::diffuseAttrib(nullptr);
    
    // delete materials 
    for (auto m : _materials) delete m;
    _materials.clear();

    // delete meshes 
    for (auto m : _meshes) delete m;
    _meshes.clear();
   
    SLMaterial::current = nullptr;
   
    // delete custom shader programs but not default shaders
    while (_programs.size() > _numProgsPreload)
    {   SLGLProgram* sp = _programs.back();
        delete sp;
        _programs.pop_back();
    }

    // delete trackers
    for (auto t : _trackers) delete t;
        _trackers.clear();

    _videoType = VT_NONE;

    _eventHandlers.clear();

    _animManager.clear();

    // reset all states
    SLGLState::getInstance()->initAll();
}
//-----------------------------------------------------------------------------
//! Processes all queued events and updates animations, AR trackers and AABBs
/*! Updates different updatables in the scene after all views got painted:
\n
\n 1) Calculate frame time
\n 2) Process queued events
\n 3) Update all animations
\n 4) Augmented Reality (AR) Tracking with the live camera
\n 5) Update AABBs
\n
A scene can be displayed in multiple views as demonstrated in the app-Viewer-Qt 
example. AR tracking is only handled on the first scene view.
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
    _elapsedTimeMS = timeMilliSec() - _lastUpdateTimeMS;
    _lastUpdateTimeMS = timeMilliSec();
     
    // Sum up all timings of all sceneviews
    SLfloat sumCullTimeMS   = 0.0f;
    SLfloat sumDraw3DTimeMS = 0.0f;
    SLfloat sumDraw2DTimeMS = 0.0f;
    SLbool renderTypeIsRT = false;
    SLbool voxelsAreShown = false;
    for (auto sv : _sceneViews)
    {   if (sv != nullptr)
        {   sumCullTimeMS   += sv->cullTimeMS();
            sumDraw3DTimeMS += sv->draw3DTimeMS();
            sumDraw2DTimeMS += sv->draw2DTimeMS();
            if (!renderTypeIsRT && sv->renderType()==RT_rt)
                renderTypeIsRT = true;
            if (!voxelsAreShown && sv->drawBit(SL_DB_VOXELS))
                voxelsAreShown = true;
        }
    }
    _cullTimesMS.set(sumCullTimeMS);
    _draw3DTimesMS.set(sumDraw3DTimeMS);
    _draw2DTimesMS.set(sumDraw2DTimeMS);

    // Calculate the frames per second metric
    _frameTimesMS.set(_elapsedTimeMS);
    _fps = 1 / _frameTimesMS.average() * 1000.0f;
    if (_fps < 0.0f) _fps = 0.0f;

    SLfloat startUpdateMS = timeMilliSec();


    //////////////////////////////
    // 2) Process queued events //
    //////////////////////////////

    // Process queued up system events and poll custom input devices
    SLbool sceneHasChanged = SLInputManager::instance().pollAndProcessEvents();


    //////////////////////////////
    // 3) Update all animations //
    //////////////////////////////

    // reset the dirty flag on all skeletons
    for(auto skeleton : _animManager.skeletons())
        skeleton->changed(false);

    sceneHasChanged |= !_stopAnimations && _animManager.update(elapsedTimeSec());
    
    // Do software skinning on all changed skeletons
    for (auto mesh : _meshes) 
    {   if (mesh->skeleton() && mesh->skeleton()->changed())
        {   mesh->transformSkin();
            sceneHasChanged = true;
        }

        // update any out of date acceleration structure for RT or if they're being rendered.
        if (renderTypeIsRT || voxelsAreShown)
            mesh->updateAccelStruct();
    }
    

    ////////////////////
    // 4) AR Tracking //
    ////////////////////
    
    if (_videoType!=VT_NONE && !SLCVCapture::lastFrame.empty())
    {   
        SLfloat trackingTimeStartMS = timeMilliSec();

        // Invalidate calibration if camera input aspect doesn't match output
        SLfloat calibWdivH = _activeCalib->imageAspectRatio();
        SLbool aspectRatioDoesNotMatch = SL_abs(_sceneViews[0]->scrWdivH() - calibWdivH) > 0.01f;
        if (aspectRatioDoesNotMatch && _activeCalib->state() == CS_calibrated)
        {   _activeCalib->clear();
        }

        stringstream ss; // info line text

        //.....................................................................
        if (_activeCalib->state() == CS_uncalibrated)
        {
            if (SL::currentSceneID == C_sceneVideoCalibrateMain ||
                SL::currentSceneID == C_sceneVideoCalibrateScnd)
            {   _activeCalib->state(CS_calibrateStream);
            } else
            {   // Changes the state to CS_guessed
                _activeCalib->createFromGuessedFOV(SLCVCapture::lastFrame.cols,
                                                   SLCVCapture::lastFrame.rows);
                _sceneViews[0]->camera()->fov(_activeCalib->cameraFovDeg());
            }
        } else //..............................................................
        if (_activeCalib->state() == CS_calibrateStream ||
            _activeCalib->state() == CS_calibrateGrab)
        {
            _activeCalib->findChessboard(SLCVCapture::lastFrame,
                                         SLCVCapture::lastFrameGray,
                                         true);
            int imgsToCap = _activeCalib->numImgsToCapture();
            int imgsCaped = _activeCalib->numCapturedImgs();

            //update info line
            if(imgsCaped < imgsToCap)
                ss << "Click on the screen to create a calibration photo. Created "
                   << imgsCaped << " of " << imgsToCap;
            else
            {   ss << "Calculating, please wait ...";
                _activeCalib->state(CS_startCalculating);
            }
            _info = ss.str();
        } else //..............................................................
        if (_activeCalib->state() == CS_startCalculating)
        {
            if (_activeCalib->calculate())
            {   _sceneViews[0]->camera()->fov(_activeCalib->cameraFovDeg());
                if (SL::currentSceneID == C_sceneVideoCalibrateMain)
                     onLoad(_sceneViews[0], C_sceneVideoTrackChessMain);
                else onLoad(_sceneViews[0], C_sceneVideoTrackChessScnd);
            }
        } else
        if (_activeCalib->state() == CS_calibrated ||
            _activeCalib->state() == CS_guessed) //............................
        {
            SLCVTrackedAruco::trackAllOnce = true;
        
            // track all trackers in the first sceneview
            for (auto tracker : _trackers)
                tracker->track(SLCVCapture::lastFrameGray,
                               SLCVCapture::lastFrame,
                               _activeCalib,
                               _showDetection,
                               _sceneViews[0]);

            // Update info text only for chessboard scene
            if (SL::currentSceneID == C_sceneVideoCalibrateMain ||
                SL::currentSceneID == C_sceneVideoCalibrateScnd ||
                SL::currentSceneID == C_sceneVideoTrackChessMain ||
                SL::currentSceneID == C_sceneVideoTrackChessScnd)
            {
                SLfloat fov = _activeCalib->cameraFovDeg();
                SLfloat err = _activeCalib->reprojectionError();
                ss << "Tracking " << (_videoType==VT_MAIN ? "main " : "scnd. ") << "camera. ";
                if (_activeCalib->state() == CS_calibrated)
                     ss << "FOV: " << fov << ", error: " << err;
                else ss << "Camera is not calibrated. A FOV is guessed of: " << fov << " degrees.";
                _info = ss.str();
            }
        } //...................................................................

        //copy image to video texture
        if(_activeCalib->state() == CS_calibrated && _activeCalib->showUndistorted())
        {
            SLCVMat undistorted;
            _activeCalib->remap(SLCVCapture::lastFrame, undistorted);

            _videoTexture.copyVideoImage(undistorted.cols,
                                         undistorted.rows,
                                         SLCVCapture::format,
                                         undistorted.data,
                                         undistorted.isContinuous(),
                                         true);
        } else
        {   _videoTexture.copyVideoImage(SLCVCapture::lastFrame.cols,
                                         SLCVCapture::lastFrame.rows,
                                         SLCVCapture::format,
                                         SLCVCapture::lastFrame.data,
                                         SLCVCapture::lastFrame.isContinuous(),
                                         true);
        }

        _trackingTimesMS.set(timeMilliSec()-trackingTimeStartMS);
    }


    /////////////////////
    // 5) Update AABBs //
    /////////////////////

    // The updateAABBRec call won't generate any overhead if nothing changed
    SLGLState::getInstance()->modelViewMatrix.identity();
    if (_root3D)
        _root3D->updateAABBRec();
    if (_root2D)
        _root2D->updateAABBRec();


    _updateTimesMS.set(timeMilliSec()-startUpdateMS);
    
    return sceneHasChanged;
}
//-----------------------------------------------------------------------------
//! SLScene::onAfterLoad gets called after onLoad
void SLScene::onAfterLoad()
{
    #ifdef SL_USES_CVCAPTURE
    if (_videoType!=VT_NONE)
    {   if (!SLCVCapture::isOpened())
        #ifdef SL_VIDEO_DEBUG
        SLCVCapture::open("../_data/videos/testvid_" + string(SL_TRACKER_IMAGE_NAME) +".mp4");
        #else
        SLCVCapture::open(0);
        #endif
    }
    #endif
}
//-----------------------------------------------------------------------------
/*!
SLScene::onRotationPYR: Event handler for rotation change of a mobile device
with Euler angles for pitch, yaw and roll. This function will only be called
in an Android or iOS project. See onRotationPYR in GLES3Activity.java in the
Android project.
This handler is only called if the flag SLScene::_usesRotation is true. If so
the mobile device turns on it's IMU sensor system. The device rotation is so
far only used in SLCamera::setViev if the cameras animation is on CA_deciveRotYUp.
If _zeroYawAfterStart is true the start yaw value is subtracted. This means
that the magnetic north will be ignored.
The angles should be:\n
Pitch from -halfpi (down)  to zero (horizontal) to +halfpi (up)\n
Yaw   from -pi     (south) to zero (north)      to +pi     (south)\n
Roll  from -halfpi (ccw)   to zero (horizontal) to +halfpi (clockwise)\n
*/
void SLScene::onRotationPYR(SLfloat pitchRAD,
                            SLfloat yawRAD,
                            SLfloat rollRAD)
{
/*
_devicePitchRAD = pitchRAD;
_deviceYawRAD   = yawRAD;
_deviceRollRAD  = rollRAD;


if (_zeroYawAtStart)
{
    if (_deviceRotStarted)
    {
        //store initial rotation as offset
        _rotationOffsetInv = _deviceRotation.inverted();
        _deviceRotStarted = false;
    }
    _deviceRotation.rotate(_rotationOffsetInv);
}
 */

    /*
    // Build quaternion from euler angles
    if (_zeroYawAtStart)
    {
        if (_deviceRotStarted)
        {   _startYawRAD = yawRAD;
            _deviceRotStarted = false;
        }
        _deviceRotation.fromEulerAngles(pitchRAD, yawRAD-_startYawRAD, rollRAD);
    }
    else
    {
        _deviceRotation.fromEulerAngles(pitchRAD, yawRAD, rollRAD);
    }
     */
}
//-----------------------------------------------------------------------------
/*! SLScene::onRotationQUAT: Event handler for rotation change of a mobile
device with rotation quaternion.
*/
void SLScene::onRotationQUAT(SLfloat quatX,
                             SLfloat quatY,
                             SLfloat quatZ,
                             SLfloat quatW)
{
    SLQuat4f quat(quatX, quatY, quatZ, quatW);
    //_deviceRotation.set(quatX, quatY, quatZ, quatW);
    //quat.toEulerAngles(_devicePitchRAD, _deviceYawRAD, _deviceRollRAD);

    _deviceRotation = quat.toMat4();
    _deviceRotation.toEulerAnglesZYX(_deviceYawRAD, _deviceRollRAD, _devicePitchRAD );
    //_deviceRotation.identity();
    //_deviceRotation.rotate( -90, 0, 0, 1);
    //_deviceRotation *= quat.toMat4();


    if (_zeroYawAtStart)
    {
        if (_deviceRotStarted)
        {
            //store initial rotation as offset
            _rotationOffsetInv = _deviceRotation.inverse();
            _deviceRotStarted = false;
        }
        //_deviceRotation.rotate(_rotationOffsetInv);
        //_deviceRotation =  _rotationOffsetInv * _deviceRotation;
    }

    //add rotation about -90 degrees around z-axis because our scene camera is
    //rotated like that with respect to the sensor coordinate system
    //_deviceRotation.rotate( 90, 0, 0, 1);
}
//-----------------------------------------------------------------------------
//! Sets the _selectedNode to the passed Node and flags it as selected
void SLScene::selectNode(SLNode* nodeToSelect)
{
    if (_selectedNode)
        _selectedNode->drawBits()->off(SL_DB_SELECTED);

    if (nodeToSelect)
    {  if (_selectedNode == nodeToSelect)
        {   _selectedNode = 0;
        } else
        {   _selectedNode = nodeToSelect;
            _selectedNode->drawBits()->on(SL_DB_SELECTED);
        }
    } else _selectedNode = 0;
    _selectedMesh = 0;
}
//-----------------------------------------------------------------------------
//! Sets the _selectedNode and _selectedMesh and flags it as selected
void SLScene::selectNodeMesh(SLNode* nodeToSelect, SLMesh* meshToSelect)
{
    if (_selectedNode)
        _selectedNode->drawBits()->off(SL_DB_SELECTED);

    if (nodeToSelect)
    {  if (_selectedNode == nodeToSelect && _selectedMesh == meshToSelect)
        {   _selectedNode = 0;
            _selectedMesh = 0;
        } else
        {   _selectedNode = nodeToSelect;
            _selectedMesh = meshToSelect;
            _selectedNode->drawBits()->on(SL_DB_SELECTED);
        }
    } else 
    {   _selectedNode = 0;
        _selectedMesh = 0;
    }
}
//-----------------------------------------------------------------------------
//! Executes a command on all sceneview
SLbool SLScene::onCommandAllSV(const SLCommand cmd)
{
    SLbool result = false;
    for(auto sv : _sceneViews)
        if (sv != nullptr)
            result = sv->onCommand(cmd) ? true : result;

    return true;
}
//-----------------------------------------------------------------------------
//! Copies the image data from a video camera into image[0] of the video texture
void SLScene::copyVideoImage(SLint width,
                             SLint height,
                             SLPixelFormat srcPixelFormat,
                             SLuchar* data,
                             SLbool isContinuous,
                             SLbool isTopLeft)
{
    _videoTexture.copyVideoImage(width, 
                                 height, 
                                 srcPixelFormat, 
                                 data, 
                                 isContinuous,
                                 isTopLeft);
}
//-----------------------------------------------------------------------------
void SLScene::onLoadAsset(SLstring assetFile, 
                          SLuint processFlags)
{
    SL::currentSceneID = C_sceneFromFile;

    // Set scene name for new scenes
    if (!_root3D)
        name(SLUtils::getFileName(assetFile));

    // Try to load assed and add it to the scene root node
    SLAssimpImporter importer;

    //////////////////////////////////////////////////////////////
    SLNode* loaded = importer.load(assetFile, true, processFlags);
    //////////////////////////////////////////////////////////////

    // Add root node on empty scene
    if (!_root3D)
    {   SLNode* scene = new SLNode("Scene");
        _root3D = scene;
    }

    // Add loaded scene
    if (loaded) 
        _root3D->addChild(loaded);

    // Add directional light if no light was in loaded asset
    if (!_lights.size())
    {   SLAABBox boundingBox = _root3D->updateAABBRec();
        SLfloat arrowLength = boundingBox.radiusWS() > FLT_EPSILON ? 
                              boundingBox.radiusWS() * 0.1f : 0.5f;
        SLLightDirect* light = new SLLightDirect(0,0,0, 
                                                 arrowLength,
                                                 1.0f, 1.0f, 1.0f);
        SLVec3f pos = boundingBox.maxWS().isZero() ? 
                      SLVec3f(1,1,1) : boundingBox.maxWS() * 1.1f;
        light->translation(pos);
        light->lookAt(pos-SLVec3f(1,1,1));
        light->attenuation(1,0,0);
        _root3D->addChild(light);
        _root3D->aabb()->reset(); // rest aabb so that it is recalculated
    }

    // call onInitialize on all scene views
    for (auto sv : _sceneViews)
    {   if (sv != nullptr)
        {   sv->onInitialize();
        }
    }
}
//-----------------------------------------------------------------------------
//! Setter for video type also sets the active calibration
/*! The SLScene instance has two video camera calibrations, one for a main camera
(SLScene::_calibMainCam) and one for the selfie camera on mobile devices
(SLScene::_calibScndCam). The member SLScene::_activeCalib references the active
one and is set by the SLScene::videoType (VT_NONE, VT_MAIN, VT_SCND) during the
scene assembly in SLScene::onLoad.
*/
void SLScene::videoType(SLVideoType vt)
{
    if (SLCVCapture::hasSecondaryCamera && vt==VT_SCND)
    {   _videoType = VT_SCND;
        _activeCalib = &_calibScndCam;
        return;
    }

    if (vt==VT_SCND)
         _videoType = VT_MAIN;
    else _videoType = vt;

    _activeCalib = &_calibMainCam;
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

    if (cams.size()==0) return nullptr;
    if (cams.size()==1) return cams[0];

    SLuint activeIndex = 0;
    for (SLuint i=0; i<cams.size(); ++i)
    {   if (cams[i] == activeSV->camera())
        {   activeIndex = i;
            break;
        }
    }

    // return next if not last else return first
    if (activeIndex < cams.size()-1)
        return cams[activeIndex+1];
    else 
        return cams[0];

}
//-----------------------------------------------------------------------------
//! Setter that turns on the device rotation sensor
void SLScene::usesRotation (SLbool use)
{
    if (!_usesRotation && use==true)
        _deviceRotStarted = true;

    _usesRotation = use;
}
//-----------------------------------------------------------------------------
