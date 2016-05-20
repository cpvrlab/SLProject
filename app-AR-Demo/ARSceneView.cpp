//#############################################################################
//  File:      ARSceneView.h
//  Purpose:   Augmented Reality Demo
//  Author:    Michael Göttlicher
//  Date:      May 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLBox.h>
#include <SLLightSphere.h>
#include <ARTracker.h>
#include <SLAssimpImporter.h>
#include <SLImage.h>
#include <SLTexFont.h>
#include <SLText.h>

#include <ARChessboardTracker.h>
#include <ARArucoTracker.h>
#include "ARSceneView.h"
#include <GLFW/glfw3.h>
#include <sstream>

#include <opencv2/highgui.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
extern GLFWwindow* window;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void SLScene::onLoad(SLSceneView* sv, SLCommand cmd)
{
    init();    

    //setup camera
    SLCamera* cam1 = new SLCamera;

    float fov = 1.0f;
    if( ARSceneView* arSV = dynamic_cast<ARSceneView*>(sv))
        fov = arSV->getCameraFov();
    cam1->fov(fov);
    cam1->clipNear(0.01);
    cam1->clipFar(10);
    //initial translation: will be overwritten as soon as first camera pose is estimated in ARTracker
    cam1->translate(0,0,0.5f);

    //set video image as backgound texture
    _background.texture(&_videoTexture, true);
    _usesVideoImage = true;

    SLLightSphere* light1 = new SLLightSphere(0.3f);
    light1->translation(0,0,10);

    SLNode* scene = new SLNode;
    scene->addChild(light1);
    scene->addChild(cam1);

    _root3D = scene;

    sv->camera(cam1);
    sv->showMenu(false);
    sv->waitEvents(false);
    sv->onInitialize();
}
//-----------------------------------------------------------------------------
ARSceneView::ARSceneView(string calibFileDir, string paramFilesDir) :
    _tracker(nullptr),
    _infoText(nullptr),
    _newMode(Idle),
    _currMode(Idle),
    _calibFileDir(calibFileDir),
    _paramFilesDir(paramFilesDir)
{
    loadCamParams(calibFileDir + "michis_calibration.xml");
}
//-----------------------------------------------------------------------------
ARSceneView::~ARSceneView()
{
    if(_tracker) delete _tracker; _tracker = nullptr;
    if(_infoText) delete _infoText; _infoText = nullptr;
}
//-----------------------------------------------------------------------------
void ARSceneView::loadCamParams(string filename)
{
    //load camera parameter
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Could not open the calibration file" << endl;
        return;
    }

    fs["camera_matrix"] >> _intrinsics;
    fs["distortion_coefficients"] >> _distortion;
    // close the input file
    fs.release();

    //calculate projection matrix
    calculateCameraFieldOfView();
}
//-----------------------------------------------------------------------------
void ARSceneView::calculateCameraFieldOfView()
{
    //calculate vertical field of view
    float fy = _intrinsics.at<double>(1,1);
    float cy = _intrinsics.at<double>(1,2);
    float fovRad = 2 * atan2( cy, fy );
    _cameraFovDeg = fovRad * SL_RAD2DEG;
}
//-----------------------------------------------------------------------------
void ARSceneView::postSceneLoad()
{
    updateInfoText();
}
//-----------------------------------------------------------------------------
void ARSceneView::loadNewFrameIntoTracker()
{
    //convert video image to cv::Mat and set into tracker
    int ocvType = -1;
    switch (_lastVideoFrame->format())
    {   case PF_luminance: ocvType = CV_8UC1; break;
        case PF_rgb: ocvType = CV_8UC3; break;
        case PF_rgba: ocvType = CV_8UC4; break;
        default: SL_EXIT_MSG("OpenCV image format not supported");
    }

    if( ocvType != -1 )
    {
        cv::Mat newImage( _lastVideoFrame->height(), _lastVideoFrame->width(), ocvType, _lastVideoFrame->data());
        _tracker->setImage(newImage);
        //cv::imwrite("newImg.png", newImage);
    }
}
//-----------------------------------------------------------------------------
void ARSceneView::preDraw()
{
    //check if mode has changed
    if(_newMode != _currMode )
    {
        if(_tracker) {
            //todo: unload old scene graph objects
            _tracker->unloadSGObjects();
            //delete tracker instance
            delete _tracker; _tracker = nullptr;
        }

        //try to init this tracker mode
        switch( _newMode )
        {
        case ARSceneViewMode::Idle:
            break;
        case ARSceneViewMode::ChessboardMode:
            //instantiate
            _tracker = new ARChessboardTracker(_intrinsics, _distortion);
            //initialize
            if(!_tracker->init(_paramFilesDir))
            {
                //init failed
                _newMode = ARSceneViewMode::ArucoMode;
                updateInfoText();
            }
            break;
        case ARSceneViewMode::ArucoMode:
            //instantiate
            _tracker = new ARArucoTracker(_intrinsics, _distortion);
            //initialize
            if(!_tracker->init(_paramFilesDir))
            {
                //init failed
                _newMode = ARSceneViewMode::Idle;
                updateInfoText();
            }
            break;
        }

        //at last set _oldMode to _curMode
        _currMode = _newMode;
    }

    if(_tracker)
    {
        //convert video image to cv::Mat and set into tracker
        loadNewFrameIntoTracker();

        if( _currMode != ARSceneViewMode::Idle || _currMode != ARSceneViewMode::CalibrationMode )
        {
            _tracker->track();
            _tracker->updateSceneView(this);
        }
    }
}
//-----------------------------------------------------------------------------
void ARSceneView::postDraw()
{
    renderText();
}
//-----------------------------------------------------------------------------
void ARSceneView::updateInfoText()
{
    if (_infoText) delete _infoText;

    SLchar m[2550];   // message character array
    m[0]=0;           // set zero length

    SLstring modes;
    modes = "Mode selection: \\n";
    modes += "c: Calibrate \\n";
    modes += "0: Tracking disabled \\n";
    modes += "1: Track chessboard \\n";
    modes += "2: Track ArUco markers \\n";

    SLstring modeName;
    switch (_newMode)
    {
    case CalibrationMode:
        modeName = "Calibration Mode";
        break;
    case Idle:
        modeName = "Tracking Disabled Mode";
        break;
    case ChessboardMode:
        modeName = "Chessboard Tracking Mode";
        break;
    case ArucoMode:
        modeName = "Aruco Tracking Mode";
        break;
    }

    sprintf(m+strlen(m), "%s", modes.c_str());

    string title = modeName;
    glfwSetWindowTitle(window, title.c_str());

    SLTexFont* f = SLTexFont::getFont(1.2f, _dpi);
    _infoText = new SLText(m, f, SLCol4f::BLACK, (SLfloat)_scrW, 1.0f);
    _infoText->translate(10.0f, -_infoText->size().y-5.0f, 0.0f, TS_object);
}
//-----------------------------------------------------------------------------
void ARSceneView::renderText()
{
    if (!_infoText)
        return;

    SLScene* s = SLScene::current;
    SLfloat w2 = (SLfloat)_scrWdiv2;
    SLfloat h2 = (SLfloat)_scrHdiv2;
    SLfloat depth = 0.9f;               // Render depth between -1 & 1

    _stateGL->depthMask(false);         // Freeze depth buffer for blending
    _stateGL->depthTest(false);         // Disable depth testing
    _stateGL->blend(true);              // Enable blending
    _stateGL->polygonLine(false);       // Only filled polygons

    // Set orthographic projection with 0,0,0 in the screen center
    _stateGL->projectionMatrix.ortho(-w2, w2,-h2, h2, 1.0f, -1.0f);

    // Set viewport over entire screen
    _stateGL->viewport(0, 0, _scrW, _scrH);

    _stateGL->modelViewMatrix.identity();
    _stateGL->modelViewMatrix.translate(-w2, h2, depth);
    _stateGL->modelViewMatrix.multiply(_infoText->om());
    _infoText->drawRec(this);

    _stateGL->blend(false);       // turn off blending
    _stateGL->depthMask(true);    // enable depth buffer writing
    _stateGL->depthTest(true);    // enable depth testing
    GET_GL_ERROR;                 // check if any OGL errors occured
}
//-----------------------------------------------------------------------------
SLbool ARSceneView::onKeyPress(const SLKey key, const SLKey mod)
{
    switch(key)
    {
    case 'C':
        _newMode = ARSceneViewMode::CalibrationMode;
        break;
    case '0':
        _newMode = ARSceneViewMode::Idle;
        break;
    case '1':
        _newMode = ARSceneViewMode::ChessboardMode;
        break;
    case '2':
        _newMode = ARSceneViewMode::ArucoMode;
        break;
    }

    updateInfoText();
}
//-----------------------------------------------------------------------------

