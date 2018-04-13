//#############################################################################
//  File:      SLImGuiInfosMapTransform.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSMAPTRANSFORM_H
#define SL_IMGUI_INFOSMAPTRANSFORM_H

#include <string>
#include <SLImGuiInfosDialog.h>

//interface
class SLTrackingInfosInterface;
class SLCVMap;

//-----------------------------------------------------------------------------
class SLImGuiInfosMapTransform : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosMapTransform(std::string name, SLCVMap* map);

    void buildInfos() override;

private:
    SLCVMap* _map = nullptr;

    float _transformationRotValue = 10.0f;;
    float _transformationTransValue = 0.1f;
    float _transformationScaleValue = 1.1f;
};

#endif //SL_IMGUI_INFOSMAPTRANSFORM_H


