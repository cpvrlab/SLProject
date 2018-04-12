//#############################################################################
//  File:      SLImGuiInfosMapNode.cpp
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
    ////keyframe infos
    //if (ImGui::CollapsingHeader("KeyFrames"))
    //{
    //    //ImGui::Text("KeyFrames");
    //    //add number of keyframes
    //    if (keyFrames) {
    //        ImGui::Text("Number of Keyframes : %d ", keyFrames->children().size());
    //    }
    //    //show keyframe scene objects
    //    if (keyFrames)
    //    {
    //        b = !keyFrames->drawBits()->get(SL_DB_HIDDEN);
    //        if (ImGui::Checkbox("Show", &b))
    //        {
    //            keyFrames->drawBits()->set(SL_DB_HIDDEN, !b);
    //            for (SLNode* child : keyFrames->children()) {
    //                if (child)
    //                    child->drawBits()->set(SL_DB_HIDDEN, !b);
    //            }
    //        }
    //    }

    //    //get keyframe database
    //    if (SLCVKeyFrameDB* kfDB = _interface->getKfDB())
    //    {
    //        //if backgound rendering is active kf images will be rendered on 
    //        //near clipping plane if kf is not the active camera
    //        b = kfDB->renderKfBackground();
    //        if (ImGui::Checkbox("Show Image", &b))
    //        {
    //            kfDB->renderKfBackground(b);
    //        }

    //        //allow SLCVCameras as active camera so that we can look through it
    //        b = kfDB->allowAsActiveCam();
    //        if (ImGui::Checkbox("Allow as Active Cam", &b))
    //        {
    //            kfDB->allowAsActiveCam(b);
    //        }
    //    }
    //    //ImGui::Separator();
    //}

    ////-------------------------------------------------------------------------
    //if (ImGui::CollapsingHeader("Alignment"))
    //{
    //    //ImGui::Text("Alignment");
    //    //slider to adjust transformation value
    //    //ImGui::SliderFloat("Value", &SLGLImGui::transformationValue, -10.f, 10.f, "%5.2f");

    //    //rotation
    //    ImGui::InputFloat("Rot. Value", &SLGLImGui::transformationRotValue, 0.1f);
    //    SLGLImGui::transformationRotValue = ImClamp(SLGLImGui::transformationRotValue, -360.0f, 360.0f);

    //    static SLfloat sp = 3; //spacing
    //    SLfloat bW = (ImGui::GetContentRegionAvailWidth() - 2 * sp) / 3;
    //    if (ImGui::Button("RotX", ImVec2(bW, 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationRotValue, SLCVTrackedRaulMur::ROT_X);
    //    } ImGui::SameLine(0.0, sp);
    //    if (ImGui::Button("RotY", ImVec2(bW, 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationRotValue, SLCVTrackedRaulMur::ROT_Y);
    //    } ImGui::SameLine(0.0, sp);
    //    if (ImGui::Button("RotZ", ImVec2(bW, 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationRotValue, SLCVTrackedRaulMur::ROT_Z);
    //    }
    //    ImGui::Separator();

    //    //translation
    //    ImGui::InputFloat("Transl. Value", &SLGLImGui::transformationTransValue, 0.1f);

    //    if (ImGui::Button("TransX", ImVec2(bW, 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationTransValue, SLCVTrackedRaulMur::TRANS_X);
    //    } ImGui::SameLine(0.0, sp);
    //    if (ImGui::Button("TransY", ImVec2(bW, 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationTransValue, SLCVTrackedRaulMur::TRANS_Y);
    //    } ImGui::SameLine(0.0, sp);
    //    if (ImGui::Button("TransZ", ImVec2(bW, 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationTransValue, SLCVTrackedRaulMur::TRANS_Z);
    //    }
    //    ImGui::Separator();

    //    //scale
    //    ImGui::InputFloat("Scale Value", &SLGLImGui::transformationScaleValue, 0.1f);
    //    SLGLImGui::transformationScaleValue = ImClamp(SLGLImGui::transformationScaleValue, 0.0f, 1000.0f);

    //    if (ImGui::Button("Scale", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //        _interface->applyTransformation(SLGLImGui::transformationScaleValue, SLCVTrackedRaulMur::SCALE);
    //    }
    //    ImGui::Separator();

    //    if (ImGui::Button("Save State", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //        _interface->saveState();
    //    }
    //}

    //ImGui::End();
}