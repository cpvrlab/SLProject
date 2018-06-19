//#############################################################################
//  File:      SLImGuiInfosTracking.cpp
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
#include <SLImGuiInfosTracking.h>
#include <SLImGuiInfosDialog.h>
#include <SLTrackingInfosInterface.h>

//-----------------------------------------------------------------------------
SLImGuiInfosTracking::SLImGuiInfosTracking(std::string name, SLTrackingInfosInterface* tracker)
    : SLImGuiInfosDialog(name),
    _interface(tracker)
{
}
//-----------------------------------------------------------------------------
void SLImGuiInfosTracking::buildInfos()
{
    //-------------------------------------------------------------------------
    //numbers
    //add tracking state
    ImGui::Text("Tracking State : %s ", _interface->getPrintableState().c_str());
    //tracking type
    ImGui::Text("Tracking Type : %s ", _interface->getPrintableType().c_str());
    //mean reprojection error
    ImGui::Text("Mean Reproj. Error : %f ", _interface->meanReprojectionError());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches : %d ", _interface->getNMapMatches());
    //L2 norm of the difference between the last and the current camera pose
    ImGui::Text("Pose Difference : %f ", _interface->poseDifference());
    ImGui::Separator();

    SLbool b;
    //-------------------------------------------------------------------------
    //keypoints infos
    if (ImGui::CollapsingHeader("KeyPoints"))
    {
        //show 2D key points in video image
        b = _interface->showKeyPoints();
        if (ImGui::Checkbox("KeyPts", &b))
        {
            _interface->showKeyPoints(b);
        }

        //show matched 2D key points in video image
        b = _interface->showKeyPointsMatched();
        if (ImGui::Checkbox("KeyPts Matched", &b))
        {
            _interface->showKeyPointsMatched(b);
        }

        ////undistort image
        //SLCVCalibration* ac = SLApplication::activeCalib;
        //b = (ac->showUndistorted() && ac->state() == CS_calibrated);
        //if (ImGui::Checkbox("Undistort Image", &b))
        //{
        //    ac->showUndistorted(b);
        //}
        //ImGui::Separator();
    }
    //-------------------------------------------------------------------------
    //mappoints infos
    if (ImGui::CollapsingHeader("MapPoints"))
    {
        //number of map points
        ImGui::Text("Count : %d ", _interface->mapPointsCount());
        //show and update all mappoints
        b = _interface->showMapPC();
        ImGui::Checkbox("Show Map Pts", &b);
        _interface->showMapPC(b);

        //show and update matches to mappoints
        b = _interface->showMatchesPC();
        if (ImGui::Checkbox("Show Matches to Map Pts", &b))
        {
            _interface->showMatchesPC(b);
        }
        //show and update local map points
        b = _interface->showLocalMapPC();
        if (ImGui::Checkbox("Show Local Map Pts", &b))
        {
            _interface->showLocalMapPC(b);
        }
    }
    //-------------------------------------------------------------------------
    //keyframe infos
    if (ImGui::CollapsingHeader("KeyFrames"))
    {
        //add number of keyframes
        ImGui::Text("Number of Keyframes : %d ", _interface->getNumKeyFrames());
        //show keyframe scene objects
        //show and update all mappoints
        b = _interface->showKeyFrames();
        ImGui::Checkbox("Show", &b);
        _interface->showKeyFrames(b);

        //if backgound rendering is active kf images will be rendered on 
        //near clipping plane if kf is not the active camera
        b = _interface->renderKfBackground();
        ImGui::Checkbox("Show Image", &b);
        _interface->renderKfBackground(b);

        //allow SLCVCameras as active camera so that we can look through it
        b = _interface->allowKfsAsActiveCam();
        ImGui::Checkbox("Allow as Active Cam", &b);
        _interface->allowKfsAsActiveCam(b);
    }
}
