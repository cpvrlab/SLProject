//#############################################################################
//  File:      NewNodeGui.cpp
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################



#include <stdafx.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include "NewNodeGui.h"
#include "NewNodeSceneView.h"

//-----------------------------------------------------------------------------
SLstring NewNodeGui::infoText = "";
//-----------------------------------------------------------------------------
void NewNodeGui::buildDemoGui(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Scene Information");
    ImGui::TextWrapped(infoText.c_str());
    ImGui::End();
}
//-----------------------------------------------------------------------------

