//#############################################################################
//  File:      SLImGuiInfosTracking.h
//  Date:      Mai 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Jan Dellsperger, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSCAMERAMOVEMENT_H
#define SL_IMGUI_INFOSCAMERAMOVEMENT_H

#include <string>
#include <SLImGuiInfosDialog.h>

#define MAX_CAM_MOVEMENT_RECORD_COUNT 100

class SLImGuiInfosCameraMovement : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosCameraMovement(string name, SLCVStateEstimator* stateEstimator);
    void buildInfos() override;

private:
    SLCVStateEstimator* _stateEstimator;
    float               tX[MAX_CAM_MOVEMENT_RECORD_COUNT];
    float               tY[MAX_CAM_MOVEMENT_RECORD_COUNT];
    float               tZ[MAX_CAM_MOVEMENT_RECORD_COUNT];
    float               rX[MAX_CAM_MOVEMENT_RECORD_COUNT];
    float               rY[MAX_CAM_MOVEMENT_RECORD_COUNT];
    float               rZ[MAX_CAM_MOVEMENT_RECORD_COUNT];
    int                 recordIndex = 0;
};

#endif SL_IMGUI_INFOSCAMERAMOVEMENT_H
