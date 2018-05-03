//#############################################################################
//  File:      SLImGuiInfosTracking.h
//  Author:    Jan Dellsperger
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSCAMERAMOVEMENT_H
#define SL_IMGUI_INFOSCAMERAMOVEMENT_H

#include <string>
#include <SLImGuiInfosDialog.h>

class SLImGuiInfosCameraMovement : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosCameraMovement(std::string name, SLCVStateEstimator* stateEstimator);
    void buildInfos() override;

private:
    SLCVStateEstimator* _stateEstimator;
};

#endif SL_IMGUI_INFOSCAMERAMOVEMENT_H
