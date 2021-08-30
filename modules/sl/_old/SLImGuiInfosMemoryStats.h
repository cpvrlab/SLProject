//#############################################################################
//  File:      SLImGuiInfosMemoryStats.h
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_MEMSTATS_H
#define SL_IMGUI_MEMSTATS_H

#include <string>
#include <SLImGuiInfosDialog.h>

class SLCVMap;

//-----------------------------------------------------------------------------
class SLImGuiInfosMemoryStats : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosMemoryStats(string name, SLCVMap* map);

    void buildInfos() override;

private:
    SLCVMap* _map = NULL;
};

#endif //SL_IMGUI_MEMSTATS_H