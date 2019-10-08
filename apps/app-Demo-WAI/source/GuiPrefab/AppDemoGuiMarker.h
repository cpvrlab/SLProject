//#############################################################################
//  File:      AppDemoGuiMarker.h
//  Author:    Luc Girod, Jan Dellsperger
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_MARKER_H
#define SL_IMGUI_MARKER_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>
#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiMarker : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiMarker(const std::string& name,
                     bool*              activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
};

#endif
