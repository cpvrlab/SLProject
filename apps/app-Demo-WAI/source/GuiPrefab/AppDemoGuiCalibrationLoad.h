//#############################################################################
//  File:      AppDemoGuiVideoStorage.h
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_CALIBRATION_LOAD_H
#define SL_IMGUI_CALIBRATION_LOAD_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>
#include <WAI.h>
#include <WAICalibration.h>
#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiCalibrationLoad : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiCalibrationLoad(const std::string& name, std::string calDir, WAI::WAI * wai,  WAICalibration* wc, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    void loadCalibration(std::string path);

    std::string              _calibrationDir;
    std::vector<std::string> _existingCalibrationNames;
    std::string              _currentItem;
    WAI::WAI*                _wai;
    WAICalibration*          _wc;
};

#endif
