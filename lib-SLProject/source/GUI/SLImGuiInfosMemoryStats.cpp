//#############################################################################
//  File:      SLImGuiInfosMemoryStats.cpp
//  Author:    Michael Goettlicher
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLImGuiInfosMemoryStats.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <SLNode.h>
#include <SLApplication.h>

//-----------------------------------------------------------------------------
SLImGuiInfosMemoryStats::SLImGuiInfosMemoryStats(std::string name)
    : SLImGuiInfosDialog(name)
{
}
//-----------------------------------------------------------------------------
void SLImGuiInfosMemoryStats::buildInfos()
{
    if (SLApplication::memStats.valid())
    {
        SLApplication::memStats.updateValue();
        //ask for the memory usage via the interface and visualize it
        double val = SLApplication::memStats.getValue();

        ImGui::Text("Stats val : %d ", val);
    }
    else
    {
        ImGui::Text("Memory statistics are invalid!");
    }
}
//-----------------------------------------------------------------------------