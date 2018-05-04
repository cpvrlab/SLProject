//#############################################################################
//  File:      SLImGuiInfosMemoryStats.h
//  Author:    Michael Goettlicher
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_MEMSTATS_H
#define SL_IMGUI_MEMSTATS_H

#include <string>
#include <SLImGuiInfosDialog.h>

//-----------------------------------------------------------------------------
class SLImGuiInfosMemoryStats : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosMemoryStats(std::string name);

    void buildInfos() override;

private:
};

#endif //SL_IMGUI_MEMSTATS_H