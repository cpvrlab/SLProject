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

#include "ARSceneView.h"
#include <GLFW/glfw3.h>
#include <sstream>

#include <opencv2/highgui.hpp>

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
        fov = arSV->tracker()->getCameraFov();
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
ARSceneView::ARSceneView() :
    _tracker(nullptr),
    _infoText(nullptr),
    _curMode(ArucoMode)
{
}
//-----------------------------------------------------------------------------
ARSceneView::~ARSceneView()
{
    if(_tracker) delete _tracker; _tracker = nullptr;
    if(_infoText) delete _infoText; _infoText = nullptr;
}
//-----------------------------------------------------------------------------
void ARSceneView::postSceneLoad()
{
    if( _tracker->getType() == ARTracker::CHESSBOARD )
    {
        SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(1.0f,0.7f,0.7f));
        SLNode* box = new SLNode("Box");

        // load coordinate axis arrows
        SLAssimpImporter importer;
        SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
        axesNode->scale(0.3);
        box->addChild(axesNode);

        float edgeLength = _tracker->getCBEdgeLengthM() * 3;
        box->addMesh(new SLBox(0.0f, 0.0f, 0.0f, edgeLength, edgeLength, edgeLength, "Box", rMat));

        SLScene::current->root3D()->addChild(box);
    }

    updateInfoText();
}
//-----------------------------------------------------------------------------
static void calcObjectMatrix(const SLMat4f& cameraObjectMat, SLMat4f& objectViewMat, SLMat4f& objectMat )
{
    //calculate objectmatrix:
    /*
     * Nomenclature:
     * T = homogenious transformation matrix
     *
     * a
     *  T = homogenious transformation matrix with subscript b and superscript a
     *   b
     *
     * Subscrips and superscripts:
     * w = world
     * o = object
     * c = camera
     *
     * c
     *  T  = Transformation of object with respect to camera coordinate system. It discribes
     *   o   the position of an object in the camera coordinate system. We get this Transformation
     *       from openCVs solvePNP function.
     *
     * w       c    -1
     *  T  = (  T  )    = Transformation of camera with respect to world coordinate system. Inversion exchanges
     *   c       w        sub- and superscript
     *
     * We can combine two or more homogenious transformations to a new one if the inner sub- and superscript
     * fit together. The resulting transformation inherits the superscrip from the left and the subscript from
     * the right transformation.
     * The following tranformation is what we want to do:
     *  w    w     c
     *   T  = T  *  T   = Transformation of object with respect to world coordinate system (object matrix)
     *    o    c     o
     */

    //new object matrix = camera object matrix * object-view matrix
    objectMat = cameraObjectMat * objectViewMat;
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
    if(_tracker)
    {
        //convert video image to cv::Mat and set into tracker
        loadNewFrameIntoTracker();

        if( _tracker->getType() == ARTracker::CHESSBOARD )
        {
            if( _tracker->trackChessboard())
            {
                //update camera with calculated view matrix
                SLMat4f vm = _tracker->getViewMatrix();
                //invert view matrix because we want to set the camera object matrix
                SLMat4f camOm = vm.inverse();
                camera()->om( camOm );
            }
        }
        else if( _tracker->getType() == ARTracker::ARUCO )
        {
            if( _tracker->trackArucoMarkers())
            {
                //container for updated nodes
                std::map<int,SLNode*> updatedNodes;

                //get new ids and transformations from tracker
                std::map<int,SLMat4f>& vms = _tracker->getArucoVMs();

                bool newNodesAdded = false;
                //for all new items
                for( const auto& pair : vms)
                {
                    int key = pair.first;
                    SLMat4f ovm = pair.second;
                    //calculate object transformation matrix
                    SLMat4f om;
                    calcObjectMatrix(_camera->om(), ovm, om );

                    //check if there is already an object for this id in existingObjects
                    auto it = _arucoNodes.find(key);
                    if( it != _arucoNodes.end()) //if object already exists
                    {
                        //set object transformation matrix
                        SLNode* node = it->second;
                        //set new object matrix
                        node->om( om );
                        //set unhidden
                        node->setDrawBitsRec( SL_DB_HIDDEN, false );

                        //add to container newObjects
                        updatedNodes.insert( /*std::pair<int,SLNode*>(key, node)*/ *it );
                        //remove it from container existing objects
                        _arucoNodes.erase(it);
                    }
                    else //object does not exist
                    {
                        //create a new object
                        float r = ( rand() % 10 + 1 ) / 10.0f;
                        float g = ( rand() % 10 + 1 ) / 10.0f;
                        float b = ( rand() % 10 + 1 ) / 10.0f;
                        SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(r,g,b));
                        stringstream ss; ss << "Box" << key;
                        SLNode* box = new SLNode( ss.str() );

                        float edgeLength = _tracker->getArucoMargerLength();
                        box->addMesh(new SLBox(-edgeLength/2, -edgeLength/2, 0.0f, edgeLength/2, edgeLength/2, edgeLength, "Box", rMat));
                        //set object transformation matrix
                        box->om( om );

                        // load coordinate axis arrows
                        SLAssimpImporter importer;
                        SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
                        axesNode->scale(0.3);
                        box->addChild(axesNode);

                        //add new object to Scene
                        SLScene::current->root3D()->addChild(box);
                        newNodesAdded = true;

                        //add to container of updated objects
                        updatedNodes.insert( std::pair<int,SLNode*>(key, box));

                        cout << "Aruco Markers: Added new object for id " << key << endl << endl;
                    }
                }

                //for all remaining objects in existingObjects
                for( auto& it : _arucoNodes )
                {
                    //hide
                    SLNode* node = it.second;
                    node->setDrawBitsRec( SL_DB_HIDDEN, true );
                    //add to updated objects
                    updatedNodes.insert( it );
                }

                //update aabbs if new nodes added
                if( newNodesAdded )
                    SLScene::current->root3D()->updateAABBRec();

                //overwrite aruco nodes
                _arucoNodes = updatedNodes;
            }
        }
    }
}
//-----------------------------------------------------------------------------
void ARSceneView::postDraw()
{
    renderText();
}
//-----------------------------------------------------------------------------
void ARSceneView::initChessboardTracking(string camParamsFilename, int boardHeight, int boardWidth,
    float edgeLengthM )
{
    //Tracking initialization
    _tracker = new ARTracker();
    //load camera parameter matrix
    _tracker->loadCamParams(camParamsFilename);
    //initialize chessboard tracker
    _tracker->initChessboard(boardWidth, boardHeight, edgeLengthM);
    //set type
    _tracker->setType(ARTracker::CHESSBOARD);
}
//-----------------------------------------------------------------------------
void ARSceneView::initArucoTracking(string camParamsFilename, int dictionaryId, float markerLength,
                         string detectParamFilename )
{
    //Tracking initialization
    _tracker = new ARTracker();
    //load camera parameter matrix
    _tracker->loadCamParams(camParamsFilename);
    //initialize aruco tracker
    _tracker->initArucoMarkerDetection(dictionaryId, markerLength, detectParamFilename );
    //set type
    _tracker->setType(ARTracker::ARUCO);
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
    switch (_curMode)
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
        _curMode = ARSceneViewMode::CalibrationMode;
        break;
    case '0':
        _curMode = ARSceneViewMode::Idle;
        break;
    case '1':
        _curMode = ARSceneViewMode::ChessboardMode;
        break;
    case '2':
        _curMode = ARSceneViewMode::ArucoMode;
        break;
    }

    updateInfoText();
}
//-----------------------------------------------------------------------------

