//#############################################################################
//  File:      SLImGuiInfosMapNodeTransform.h
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Jan Dellsperger, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSMAPNODETRANSFORM_H
#define SL_IMGUI_INFOSMAPNODETRANSFORM_H

#include <SLImGuiInfosDialog.h>
#include <string>

class SLCVMapTracking;
class SLCVMapNode;

//-----------------------------------------------------------------------------
class SLImGuiInfosMapNodeTransform : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosMapNodeTransform(
      string           name,
      SLCVMapNode*     mapNode,
      SLCVMapTracking* tracking);

    void buildInfos() override;

private:
    float _transformationRotValue   = 10.0f;
    float _transformationTransValue = 1.0f;
    float _transformationScaleValue = 1.2f;

    SLCVMapNode*     _mapNode  = nullptr;
    SLCVMapTracking* _tracking = nullptr;
};
//-----------------------------------------------------------------------------
#endif //SL_IMGUI_INFOSMAPNODETRANSFORM_H
