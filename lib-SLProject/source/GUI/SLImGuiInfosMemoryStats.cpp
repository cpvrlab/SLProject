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

        ImGui::Text("freeMemoryRT:      %d ", SLApplication::memStats._freeMemoryRT);
        ImGui::Text("totalMemoryRT:     %d ", SLApplication::memStats._totalMemoryRT);
        ImGui::Text("maxMemoryRT:       %d ", SLApplication::memStats._maxMemoryRT);
        ImGui::Text("usedMemInMB:       %d ", SLApplication::memStats._usedMemInMB);
        ImGui::Text("maxHeapSizeInMB:   %d ", SLApplication::memStats._maxHeapSizeInMB);
        ImGui::Text("availHeapSizeInMB: %d ", SLApplication::memStats._availHeapSizeInMB);
        ImGui::Separator();
        ImGui::Text("availMemoryAM:     %d ", SLApplication::memStats._availMemoryAM);
        ImGui::Text("totalMemoryAM:     %d ", SLApplication::memStats._totalMemoryAM);
        ImGui::Text("thresholdAM:       %d ", SLApplication::memStats._thresholdAM);
        ImGui::Text("availableMegs:     %f ", SLApplication::memStats._availableMegs);
        ImGui::Text("percentAvail:      %f ", SLApplication::memStats._percentAvail);
        ImGui::Text("lowMemoryAM:       %b ", SLApplication::memStats._lowMemoryAM);
    }
    else
    {
        ImGui::Text("Memory statistics are invalid!");
    }
}
//-----------------------------------------------------------------------------