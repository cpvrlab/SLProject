//#############################################################################
//  File:      SLImGuiInfosTracking.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_INFOSTRACKING_H
#define SL_IMGUI_INFOSTRACKING_H

#include <string>
#include <SLImGuiInfosDialog.h>

class SLTrackingInfosInterface;
class SLCVMapNode;

//-----------------------------------------------------------------------------
class SLImGuiInfosTracking : public SLImGuiInfosDialog
{
public:
    SLImGuiInfosTracking(string name, SLTrackingInfosInterface* tracker, SLCVMapNode* mapNode);

    void buildInfos() override;

private:
    SLTrackingInfosInterface* _interface = nullptr;
    SLCVMapNode*              _mapNode   = nullptr;

    int _minNumCovisibleMapPts = 0;
};

#endif SL_IMGUI_INFOSTRACKING_H
