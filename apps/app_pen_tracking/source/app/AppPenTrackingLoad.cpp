//#############################################################################
//  File:      AppPenTrackingGui.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <GlobalTimer.h>

#include <CVCapture.h>
#include <cv/CVTrackedAruco.h>
#include <cv/CVTrackedChessboard.h>
#include <cv/CVMultiTracker.h>
#include <cv/CVCalibrationEstimator.h>

#include <SLAlgo.h>
#include <AppDemo.h>
#include <SLAssimpImporter.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLBox.h>
#include <SLCoordAxis.h>
#include <SLLightSpot.h>
#include <SLSphere.h>
#include <SLText.h>
#include <SLTexColorLUT.h>
#include <SLProjectScene.h>
#include <SLGLProgramManager.h>
#include <Instrumentor.h>
#include <app/AppPenTrackingGui.h>
#include <SLNodeLOD.h>

#include <app/AppPenTracking.h>
#include "TrackedPen.h"

//-----------------------------------------------------------------------------
void appDemoLoadScene(SLProjectScene* s, SLSceneView* sv, SLSceneID sceneID)
{
    PROFILE_FUNCTION();

    s->assetManager((SLAssetManager*)s);

    SLfloat startLoadMS = GlobalTimer::timeMS();

    // Reset non CVTracked and CVCapture infos
    CVTracked::resetTimes();                   // delete all tracker times
    CVCapture::instance()->videoType(VT_NONE); // turn off any video

    // Reset asset pointer from previous scenes
    for (const auto& tracker : AppPenTracking::instance().trackers()) delete tracker.second;
    AppPenTracking::instance().trackers().clear();
    AppPenTracking::instance().videoTexture = nullptr; // The video texture will be deleted by scene uninit
    AppPenTracking::instance().trackedNode  = nullptr; // The tracked node will be deleted by scene uninit

    AppDemo::sceneID = sceneID;

    SLstring texPath    = AppDemo::texturePath;
    SLstring dataPath   = AppDemo::dataPath;
    SLstring modelPath  = AppDemo::modelPath;
    SLstring shaderPath = AppDemo::shaderPath;

    // reset existing sceneviews
    for (auto* sceneview : AppDemo::sceneViews)
        sceneview->unInit();

    // Initialize all preloaded stuff from SLScene
    s->init();

    // Deactivate in general the device sensors
    AppDemo::devRot.init();
    AppDemo::devLoc.init();

    s->eventHandlers().push_back(&AppPenTracking::instance().arucoPen());
    CVCapture::instance()->activeCamera = &AppPenTracking::instance().currentCaptureProvider()->camera();

    if (sceneID == SID_VideoTrackChessMain ||
        sceneID == SID_VideoCalibrateMain) //.................................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateTracking by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        The chessboard marker used in these scenes is also used for the camera
        calibration. The different calibration state changes are also handled in
        AppDemoVideo::onUpdateVideo.
        */

        // Setup here only the requested scene.
        if (sceneID == SID_VideoTrackChessMain)
        {
            CVCapture::instance()->videoType(VT_MAIN);
            s->name("Track Chessboard (main cam.)");
        }
        else if (sceneID == SID_VideoCalibrateMain)
        {
            AppPenTracking::instance().calibrator().reset();
            CVCapture::instance()->videoType(VT_MAIN);
            s->name("Calibrate Main Cam.");
        }

        // Create video texture on global pointer updated in AppDemoVideo
        AppPenTracking::instance().videoTexture = new SLGLTexture(s, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* yellow = new SLMaterial(s, "mY", SLCol4f(1, 1, 0, 0.5f));

        // set the edge length of a chessboard square
        SLfloat e1 = 0.028f;
        SLfloat e3 = e1 * 3.0f;
        SLfloat e9 = e3 * 3.0f;

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node
        SLCamera* cam1 = new SLCamera();
        cam1->name("camera node");
        cam1->translation(0, 0, 5);
        cam1->lookAt(0, 0, 0);
        cam1->focalDist(5);
        cam1->clipFar(10);
        cam1->fov(CVCapture::instance()->activeCamera->calibration.cameraFovVDeg());
        cam1->background().texture(AppPenTracking::instance().videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, e1 * 0.5f);
        light1->translate(e9, e9, e9);
        light1->name("light node");
        scene->addChild(light1);

        // Build mesh & node
        if (sceneID == SID_VideoTrackChessMain)
        {
            SLBox*  box     = new SLBox(s, 0.0f, 0.0f, 0.0f, e3, e3, e3, "Box", yellow);
            SLNode* boxNode = new SLNode(box, "Box Node");
            boxNode->setDrawBitsRec(SL_DB_CULLOFF, true);
            SLNode* axisNode = new SLNode(new SLCoordAxis(s), "Axis Node");
            axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
            axisNode->scale(e3);
            boxNode->addChild(axisNode);
            scene->addChild(boxNode);
        }

        // Create OpenCV Tracker for the camera node for AR camera.
        CVTrackedChessboard* tracker = new CVTrackedChessboard(AppDemo::calibIniPath);
        tracker->drawDetection(true);
        AppPenTracking::instance().trackers().insert({AppPenTracking::instance().currentCaptureProvider(), tracker});
        AppPenTracking::instance().trackedNode = cam1;

        // pass the scene group as root node
        s->root3D(scene);

        // Set active camera
        sv->camera(cam1);
        sv->doWaitOnIdle(false);
    }
    else if (sceneID == SID_VideoTrackArucoCubeMain) //............................................
    {
        /*
        The tracking of markers is done in AppDemoVideo::onUpdateVideo by calling the specific
        CVTracked::track method. If a marker was found it overwrites the linked nodes
        object matrix (SLNode::_om). If the linked node is the active camera the found
        transform is additionally inversed. This would be the standard augmented realtiy
        use case.
        */

        CVCapture::instance()->videoType(VT_MAIN);
        s->name("Track Aruco Cube (main cam.)");
        s->info("Hold the Aruco Cube into the field of view of the main camera. You can find the Aruco markers in the file data/Calibrations. Press F6 to print the ArUco pen position and measure distances");

        // Create video texture on global pointer updated in AppDemoVideo
        AppPenTracking::instance().videoTexture = new SLGLTexture(s, texPath + "LiveVideoError.png", GL_LINEAR, GL_LINEAR);

        // Material
        SLMaterial* penTipMaterial = new SLMaterial(s, "Pen Tip Material", SLCol4f(1.0f, 1.0f, 0.0f, 0.5f));
        SLMaterial* penMaterial    = new SLMaterial(s, "Pen Material", SLCol4f(0.3f, 0.1f, 1.0f, 0.25f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0, 0, 1);
        cam1->lookAt(0, 0, 0);
        cam1->fov(36); // FIXME: Hardcoded FOV for Logitech 1080p webcam
        cam1->clipNear(0.001f);
        cam1->clipFar(10.0f);
        cam1->focalDist(1.0f);
        cam1->background().texture(AppPenTracking::instance().videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.02f);
        light1->translation(0.12f, 0.12f, 0.12f);
        light1->name("light node");
        light1->setDrawBitsRec(SL_DB_HIDDEN, true);
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen = CVTrackedAruco::params.edgeLength;
        SLfloat he      = edgeLen / 2;

        float tipOffset = 0.147f - 0.025f + 0.002f;
        float tiphe     = 0.002f;

        SLAssimpImporter importer;

        SLNode* penNode = importer.load(s->animManager(),
                                        s,
                                        modelPath + "DAE/ArucoPen/ArucoPen.dae",
                                        texPath,
                                        true,
                                        true,
                                        penMaterial);

        scene->addChild(penNode);

        SLMesh* tipMesh = new SLBox(s, -tiphe, -tiphe - tipOffset, -tiphe, tiphe, tiphe - tipOffset, tiphe, "Pen Tip", penTipMaterial);
        SLNode* tipNode = new SLNode(tipMesh, "Pen Tip Node");
        penNode->addChild(tipNode);

        SLNode* axisNode = new SLNode(new SLCoordAxis(s), "Axis Node");
        axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axisNode->scale(edgeLen);
        // scene->addChild(axisNode);

        CVTrackedAruco::params.filename = "aruco_cube_detector_params.yml";

        for (CVCaptureProvider* provider : AppPenTracking::instance().captureProviders())
        {
            CVTrackedArucoCube* tracker = new CVTrackedArucoCube(AppDemo::calibIniPath, 0.05f);
            tracker->drawDetection(true);
            AppPenTracking::instance().trackers().insert({provider, tracker});
        }

        AppPenTracking::instance().trackedNode = penNode;

        s->root3D(scene);
        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        sv->doFrustumCulling(false);
    }
    else if (sceneID == SID_VirtualTrackedPen) //............................................
    {
        CVCapture::instance()->videoType(VT_MAIN);
        s->name("Track Aruco Cube (Virtual)");
        s->info("Hold the Aruco Cube into the field of view of the main camera. You can find the Aruco markers in the file data/Calibrations. Press F6 to print the ArUco pen position and measure distances");

        // Material
        SLMaterial* penTipMaterial = new SLMaterial(s, "Pen Tip Material", SLCol4f(1.0f, 1.0f, 0.0f, 0.5f));
        SLMaterial* penMaterial    = new SLMaterial(s, "Pen Material", SLCol4f(0.3f, 0.1f, 1.0f, 0.25f));

        // Create a scene group node
        SLNode* scene = new SLNode("scene node");

        // Create a camera node 1
        SLCamera* cam1 = new SLCamera("Camera 1");
        cam1->translation(0.8f, 0.5f, 0.8f);
        cam1->lookAt(0, 0, 0);
        cam1->fov(36); // FIXME: Hardcoded FOV for Logitech 1080p webcam
        cam1->clipNear(0.001f);
        cam1->clipFar(10.0f);
        cam1->focalDist(cam1->om().translation().length());
        cam1->background().texture(AppPenTracking::instance().videoTexture);
        cam1->setInitialState();
        cam1->devRotLoc(&AppDemo::devRot, &AppDemo::devLoc);
        scene->addChild(cam1);

        // Create a light source node
        SLLightSpot* light1 = new SLLightSpot(s, s, 0.02f);
        light1->translation(1.2f, 1.2f, 1.2f);
        light1->name("light node");
        light1->setDrawBitsRec(SL_DB_HIDDEN, true);
        scene->addChild(light1);

        // Get the half edge length of the aruco marker
        SLfloat edgeLen   = CVTrackedAruco::params.edgeLength;
        float   tipOffset = AppPenTracking::instance().arucoPen().length();
        float   tiphe     = 0.002f;

        SLAssimpImporter importer;

        SLNode* chessboardNode = importer.load(s->animManager(),
                                               s,
                                               modelPath + "DAE/Chessboard/Chessboard60.dae",
                                               texPath);
        chessboardNode->drawBits()->on(SL_DB_CULLOFF);
        scene->addChild(chessboardNode);

        SLNode* penNode = importer.load(s->animManager(),
                                        s,
                                        modelPath + "DAE/ArucoPen/ArucoPen.dae",
                                        texPath);

        scene->addChild(penNode);

        SLMesh* tipMesh = new SLBox(s, -tiphe, -tiphe - tipOffset, -tiphe, tiphe, tiphe - tipOffset, tiphe, "Pen Tip", penTipMaterial);
        SLNode* tipNode = new SLNode(tipMesh, "Pen Tip Node");
//        penNode->addChild(tipNode);

        SLNode* axisNode = new SLNode(new SLCoordAxis(s), "Axis Node");
        axisNode->setDrawBitsRec(SL_DB_MESHWIRED, false);
        axisNode->scale(edgeLen);
        // scene->addChild(axisNode);

        CVTrackedAruco::params.filename = "aruco_cube_detector_params.yml";

        for (CVCaptureProvider* provider : AppPenTracking::instance().captureProviders())
        {
            CVTrackedArucoCube* tracker = new CVTrackedArucoCube(AppDemo::calibIniPath, 0.05f);
            tracker->drawDetection(true);
            AppPenTracking::instance().trackers().insert({provider, tracker});
        }

        AppPenTracking::instance().trackedNode = penNode;

        s->root3D(scene);
        sv->camera(cam1);
        sv->doWaitOnIdle(false);
        sv->doFrustumCulling(false);
    }

    ////////////////////////////////////////////////////////////////////////////
    // call onInitialize on all scene views to init the scenegraph and stats
    for (auto* sceneView : AppDemo::sceneViews)
        if (sceneView != nullptr)
            sceneView->onInitialize();

    //    if (CVCapture::instance()->videoType() != VT_NONE)
    //    {
    //        if (sv->viewportSameAsVideo())
    //        {
    //            // Pass a negative value to the start function, so that the
    //            // viewport aspect ratio can be adapted later to the video aspect.
    //            // This will be known after start.
    //            CVCapture::instance()->start(-1.0f);
    //            SLVec2i videoAspect;
    //            videoAspect.x = CVCapture::instance()->captureSize.width;
    //            videoAspect.y = CVCapture::instance()->captureSize.height;
    //            sv->setViewportFromRatio(videoAspect, sv->viewportAlign(), true);
    //        }
    //        else
    //            CVCapture::instance()->start(sv->viewportWdivH());
    //    }

    s->loadTimeMS(GlobalTimer::timeMS() - startLoadMS);
}
//-----------------------------------------------------------------------------
