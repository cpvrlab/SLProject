//#############################################################################
//  File:      AppDemoGuiInfosTracking.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <AppWAI.h>
#include <AppDemoGuiInfosTracking.h>

//-----------------------------------------------------------------------------
AppDemoGuiInfosTracking::AppDemoGuiInfosTracking(std::string        name,
                                                 WAI::ModeOrbSlam2* mode,
                                                 bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _mode(mode)
{
    _minNumCovisibleMapPts = WAIApp::minNumOfCovisibles;
}
//-----------------------------------------------------------------------------
void AppDemoGuiInfosTracking::buildInfos(SLScene* s, SLSceneView* sv)
{
    //-------------------------------------------------------------------------

    ImGui::Begin("Tracking Informations", _activator, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    //numbers
    //add tracking state
    ImGui::Text("Tracking State : %s ", _mode->getPrintableState().c_str());
    //tracking type
    ImGui::Text("Tracking Type : %s ", _mode->getPrintableType().c_str());
    //mean reprojection error
    ImGui::Text("Mean Reproj. Error : %f ", _mode->getMeanReprojectionError());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches : %d ", _mode->getNMapMatches());
    //L2 norm of the difference between the last and the current camera pose
    ImGui::Text("Pose Difference : %f ", _mode->poseDifference());
    ImGui::Separator();

    bool b;
    //-------------------------------------------------------------------------
    //keypoints infos
    if (ImGui::CollapsingHeader("KeyPoints"))
    {
        //show 2D key points in video image
        b = WAIApp::showKeyPoints;
        if (ImGui::Checkbox("KeyPts", &b))
        {
            WAIApp::showKeyPoints;
        }

        //show matched 2D key points in video image
        b = WAIApp::showKeyPointsMatched;
        if (ImGui::Checkbox("KeyPts Matched", &b))
        {
            WAIApp::showKeyPointsMatched = b;
        }
    }
    //-------------------------------------------------------------------------
    //mappoints infos
    if (ImGui::CollapsingHeader("MapPoints"))
    {
        //number of map points
        ImGui::Text("Count : %d ", _mode->getMapPointCount());
        //show and update all mappoints
        b = WAIApp::showMapPC;
        ImGui::Checkbox("Show Map Pts", &b);
        WAIApp::showMapPC = b;

        //show and update matches to mappoints
        b = WAIApp::showMatchesPC;
        if (ImGui::Checkbox("Show Matches to Map Pts", &b))
        {
            WAIApp::showMatchesPC = b;
        }
        //show and update local map points
        b = WAIApp::showLocalMapPC;
        if (ImGui::Checkbox("Show Local Map Pts", &b))
        {
            WAIApp::showLocalMapPC = b;
        }
    }
    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("KeyFrames"))
    {
        //add number of keyframes
        ImGui::Text("Number of Keyframes : %d ", _mode->getKeyFrameCount());
        //show keyframe scene objects
        //show and update all mappoints
        b = WAIApp::showKeyFrames;
        ImGui::Checkbox("Show", &b);
        WAIApp::showKeyFrames = b;

        //if backgound rendering is active kf images will be rendered on
        //near clipping plane if kf is not the active camera
        b = WAIApp::renderKfBackground;
        ImGui::Checkbox("Show Image", &b);
        WAIApp::renderKfBackground = b;

        //allow SLCVCameras as active camera so that we can look through it
        b = WAIApp::allowKfsAsActiveCam;
        ImGui::Checkbox("Allow as Active Cam", &b);
        WAIApp::allowKfsAsActiveCam = b;
    }

    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("Graph"))
    {
        //covisibility graph
        b = WAIApp::showCovisibilityGraph;
        ImGui::Checkbox("Show Covisibility (100 common KPts)", &b);
        WAIApp::showCovisibilityGraph = b;
        if (b)
        {
            //Definition of minimum number of covisible map points
            if (ImGui::InputInt("Min. covis. map pts", &_minNumCovisibleMapPts, 10, 0))
            {
                WAIApp::minNumOfCovisibles = (_minNumCovisibleMapPts);
            }
        }
        //spanning tree
        b = WAIApp::showSpanningTree;
        ImGui::Checkbox("Show spanning tree", &b);
        WAIApp::showSpanningTree = b;
        //loop edges
        b = WAIApp::showLoopEdges;
        ImGui::Checkbox("Show loop edges", &b);
        WAIApp::showLoopEdges = b;
    }
    ImGui::End();
}
