//#############################################################################
//  File:      SLImGuiInfosMemoryStats.cpp
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLImGuiInfosMemoryStats.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <SLNode.h>
#include <AppDemo.h>
#include <SLCVMap.h>

//-----------------------------------------------------------------------------
SLImGuiInfosMemoryStats::SLImGuiInfosMemoryStats(string name, SLCVMap* map)
  : SLImGuiInfosDialog(name),
    _map(map)
{
}
//-----------------------------------------------------------------------------
void SLImGuiInfosMemoryStats::buildInfos()
{
    if (AppDemo::memStats.valid())
    {
        AppDemo::memStats.updateValue();

        ImGui::Text("freeMemoryRT:      %d ", AppDemo::memStats._freeMemoryRT);
        ImGui::Text("totalMemoryRT:     %d ", AppDemo::memStats._totalMemoryRT);
        ImGui::Text("maxMemoryRT:       %d ", AppDemo::memStats._maxMemoryRT);
        ImGui::Text("usedMemInMB:       %d ", AppDemo::memStats._usedMemInMB);
        ImGui::Text("maxHeapSizeInMB:   %d ", AppDemo::memStats._maxHeapSizeInMB);
        ImGui::Text("availHeapSizeInMB: %d ", AppDemo::memStats._availHeapSizeInMB);
        ImGui::Separator();
        ImGui::Text("availMemoryAM:     %d ", AppDemo::memStats._availMemoryAM);
        ImGui::Text("totalMemoryAM:     %d ", AppDemo::memStats._totalMemoryAM);
        ImGui::Text("thresholdAM:       %d ", AppDemo::memStats._thresholdAM);
        ImGui::Text("availableMegs:     %f ", AppDemo::memStats._availableMegs);
        ImGui::Text("percentAvail:      %f ", AppDemo::memStats._percentAvail);
        ImGui::Text("lowMemoryAM:       %d ", AppDemo::memStats._lowMemoryAM);
        ImGui::Separator();
    }
    else
    {
        ImGui::Text("Memory statistics are invalid!");
    }

    if (_map)
    {
        double size = (double)_map->getSizeOf() / 1048576L;
        ImGui::Text("map size in MB     %f", size);
    }
}
//-----------------------------------------------------------------------------