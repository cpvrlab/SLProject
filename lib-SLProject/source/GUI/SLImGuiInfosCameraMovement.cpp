//#############################################################################
//  File:      SLImGuiInfosTracking.cpp
//  Author:    Jan Dellsperger
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <SLCVStateEstimator.h>
#include <SLImGuiInfosCameraMovement.h>
#include <SLImGuiInfosDialog.h>
#include <SLTrackingInfosInterface.h>

SLImGuiInfosCameraMovement::SLImGuiInfosCameraMovement(std::string name,
                                                       SLCVStateEstimator* stateEstimator)
    : SLImGuiInfosDialog(name),
      _stateEstimator(stateEstimator)
{
}

void SLImGuiInfosCameraMovement::buildInfos()
{
    SLVec3f dT = _stateEstimator->dT();
    ImGui::Text("dT : x - %f, y - %f, z - %f", dT.x, dT.y, dT.z);
    
    SLVec3f dR = _stateEstimator->dR();
    ImGui::Text("dR : x - %f, y - %f, z - %f", dR.x, dT.y, dT.z);
}
