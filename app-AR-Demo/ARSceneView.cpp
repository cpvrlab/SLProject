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
#include <SLTracker.h>
#include <SLAssimpImporter.h>

#include "ARSceneView.h"
#include <GLFW/glfw3.h>

//-----------------------------------------------------------------------------
extern GLFWwindow* window;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void SLScene::onLoad(SLSceneView* sv, SLCommand cmd)
{
    init();    

    //setup camera
    SLCamera* cam1 = new SLCamera;
    //todo: auto fov calculation
    float fov = sv->tracker()->getCameraFov();
    cam1->fov(fov);
    cam1->clipNear(0.01);
    cam1->clipFar(10);
    //initial translation: will be overwritten as soon as first camera pose is estimated in SLTracker
    cam1->translate(0,0,0.5f);

    //set video image as backgound texture
    _background.texture(&_videoTexture, true);
    _usesVideoImage = true;

    SLLightSphere* light1 = new SLLightSphere(0.3f);
    light1->translation(0,0,10);

    SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(1.0f,0.7f,0.7f));
    SLNode* box = new SLNode("Box");

    // load coordinate axis arrows
    SLAssimpImporter importer;
    SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
    axesNode->scale(0.3);

    float edgeLength = 0.105f;
    box->addMesh(new SLBox(0.0f, 0.0f, 0.0f, edgeLength, edgeLength, edgeLength, "Box", rMat));

    SLNode* scene = new SLNode;
    scene->addChild(light1);
    scene->addChild(cam1);
    scene->addChild(box);
    scene->addChild(axesNode);
    
    _root3D = scene;

    sv->camera(cam1);
    sv->showMenu(false);
    sv->waitEvents(false);
    sv->onInitialize();
}
//-----------------------------------------------------------------------------
ARSceneView::ARSceneView(string camParamsFilename, int boardHeight, int boardWidth,
    float edgeLengthM )
{
    //Tracking initialization
    _tracker = new SLTracker();
    //load camera parameter matrix
    _tracker->loadCamParams(camParamsFilename);
    //initialize chessboard tracker
    _tracker->initChessboard(boardWidth, boardHeight, edgeLengthM);
}
//-----------------------------------------------------------------------------
ARSceneView::~ARSceneView()
{
    if(_tracker) delete _tracker; _tracker = nullptr;
}
