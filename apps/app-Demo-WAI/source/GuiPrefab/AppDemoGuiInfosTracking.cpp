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
AppDemoGuiInfosTracking::AppDemoGuiInfosTracking(std::string name,
                                                 //WAI::ModeOrbSlam2* mode,
                                                 GUIPreferences& preferences)
  : AppDemoGuiInfosDialog(name, &preferences.showInfosTracking),
    _prefs(preferences)
{
    _minNumCovisibleMapPts = _prefs.minNumOfCovisibles;
}
//-----------------------------------------------------------------------------
void AppDemoGuiInfosTracking::buildInfos(SLScene* s, SLSceneView* sv)
{
    //-------------------------------------------------------------------------

    ImGui::Begin("Tracking Informations", _activator, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    //numbers
    //add tracking state
    ImGui::Text("Tracking State : %s ", WAIApp::mode->getPrintableState().c_str());
    //tracking type
    ImGui::Text("Tracking Type : %s ", WAIApp::mode->getPrintableType().c_str());
    //mean reprojection error
    ImGui::Text("Mean Reproj. Error : %f ", WAIApp::mode->getMeanReprojectionError());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches : %d ", WAIApp::mode->getNMapMatches());
    //L2 norm of the difference between the last and the current camera pose
    ImGui::Text("Pose Difference : %f ", WAIApp::mode->poseDifference());
    ImGui::Separator();

    bool b;
    //-------------------------------------------------------------------------
    //keypoints infos
    if (ImGui::CollapsingHeader("KeyPoints"))
    {
        //show 2D key points in video image
        b = _prefs.showKeyPoints;
        if (ImGui::Checkbox("KeyPts", &b))
        {
            _prefs.showKeyPoints = b;
        }

        //show matched 2D key points in video image
        b = _prefs.showKeyPointsMatched;
        if (ImGui::Checkbox("KeyPts Matched", &b))
        {
            _prefs.showKeyPointsMatched = b;
        }
    }
    //-------------------------------------------------------------------------
    //mappoints infos
    if (ImGui::CollapsingHeader("MapPoints"))
    {
        //number of map points
        ImGui::Text("Count : %d ", WAIApp::mode->getMapPointCount());
        //show and update all mappoints
        b = _prefs.showMapPC;
        ImGui::Checkbox("Show Map Pts", &b);
        _prefs.showMapPC = b;

        //show and update matches to mappoints
        b = _prefs.showMatchesPC;
        if (ImGui::Checkbox("Show Matches to Map Pts", &b))
        {
            _prefs.showMatchesPC = b;
        }
        //show and update local map points
        b = _prefs.showLocalMapPC;
        if (ImGui::Checkbox("Show Local Map Pts", &b))
        {
            _prefs.showLocalMapPC = b;
        }
    }
    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("KeyFrames"))
    {
        //add number of keyframes
        ImGui::Text("Number of Keyframes : %d ", WAIApp::mode->getKeyFrameCount());
        //show keyframe scene objects
        //show and update all mappoints
        b = _prefs.showKeyFrames;
        ImGui::Checkbox("Show", &b);
        _prefs.showKeyFrames = b;

        //if backgound rendering is active kf images will be rendered on
        //near clipping plane if kf is not the active camera
        b = _prefs.renderKfBackground;
        ImGui::Checkbox("Show Image", &b);
        _prefs.renderKfBackground = b;

        //allow SLCVCameras as active camera so that we can look through it
        b = _prefs.allowKfsAsActiveCam;
        ImGui::Checkbox("Allow as Active Cam", &b);
        _prefs.allowKfsAsActiveCam = b;
    }

    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("Graph"))
    {
        //covisibility graph
        b = _prefs.showCovisibilityGraph;
        ImGui::Checkbox("Show Covisibility (100 common KPts)", &b);
        _prefs.showCovisibilityGraph = b;
        if (b)
        {
            //Definition of minimum number of covisible map points
            if (ImGui::InputInt("Min. covis. map pts", &_minNumCovisibleMapPts, 10, 0))
            {
                _prefs.minNumOfCovisibles = (_minNumCovisibleMapPts);
            }
        }
        //spanning tree
        b = _prefs.showSpanningTree;
        ImGui::Checkbox("Show spanning tree", &b);
        _prefs.showSpanningTree = b;
        //loop edges
        b = _prefs.showLoopEdges;
        ImGui::Checkbox("Show loop edges", &b);
        _prefs.showLoopEdges = b;
    }
    ImGui::End();
}
