//#############################################################################
//  File:      SLImGuiInfosTracking.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSTRACKING_H
#define SL_IMGUI_INFOSTRACKING_H

#include <string>
#include <SLImGuiInfosDialog.h>

//interface
class SLTrackingInfosInterface;

//-----------------------------------------------------------------------------
class SLImGuiInfosTracking : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosTracking(std::string name, SLTrackingInfosInterface* tracker);

    void buildInfos() override;

private:
    SLTrackingInfosInterface* _interface = nullptr;
};

#endif //SL_IMGUI_INFOSTRACKING_H

