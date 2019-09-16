//#############################################################################
//  File:      AppDemoGuiVideoStorage.h
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_VIDEOLOAD_H
#define SL_IMGUI_VIDEOLOAD_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>
#include <WAI.h>
#include <WAICalibration.h>
#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiVideoLoad : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiVideoLoad(const std::string& name, std::string videoDir, std::string calibDir, WAICalibration* wc, WAI::WAI* wai, bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    void loadVideo(std::string videoFileName, std::string path);

    std::string              _videoDir;
    std::string              _calibDir;
    std::vector<std::string> _existingVideoNames;
    std::string              _currentItem;
    WAI::WAI*                _wai;
    WAICalibration*          _wc;
};

#endif
