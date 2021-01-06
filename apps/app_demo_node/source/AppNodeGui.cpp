//#############################################################################
//  File:      AppNodeGui.cpp
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include "AppNodeGui.h"
#include "AppNodeSceneView.h"
#include <SLScene.h>
#include <SLSceneView.h>
#include <imgui.h>
//-----------------------------------------------------------------------------
SLstring AppNodeGui::infoText = "";
//-----------------------------------------------------------------------------
//! Creates the ImGui UI.
/*! This function must be passed as void* pointer to the slCreateSceneView
function. It is called in SLSceneView::onPaint for each frame.
*/
void AppNodeGui::build(SLScene* s, SLSceneView* sv)
{
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::Begin("Scene Information",
                 0,
                 ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_AlwaysAutoResize |
                   ImGuiWindowFlags_NoMove);
    ImGui::TextUnformatted(infoText.c_str());
    ImGui::End();
}
//-----------------------------------------------------------------------------
