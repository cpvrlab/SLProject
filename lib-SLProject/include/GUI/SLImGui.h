//#############################################################################
//  File:      SLImGuiTrackedMapping.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_TRACKEDMAPPING_H
#define SL_IMGUI_TRACKEDMAPPING_H

#include <SLImGuiInfosDialog.h>

class SLCVTrackedMapping;

//-----------------------------------------------------------------------------
class SLImGuiTrackedMapping : public SLImGuiInfosDialog
{
public:
    SLImGuiTrackedMapping(std::string name, SLCVTrackedMapping* mappingTracker);

    void buildInfos() override;

private:
    SLCVTrackedMapping* _mappingTracker = nullptr;
};

#endif //SL_IMGUI_TRACKEDMAPPING_H