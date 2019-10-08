//#############################################################################
//  File:      AppDemoGuiSlamLoad.h
//  Author:    Luc Girod, Jan Dellsperger
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_SLAMLOAD_H
#define SL_IMGUI_SLAMLOAD_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>
#include <WAICalibration.h>
#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiSlamLoad : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiSlamLoad(const std::string& name,
                       WAICalibration*    wc,
                       bool*              activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    void loadFileNamesInVector(std::string               directory,
                               std::vector<std::string>& fileNames,
                               std::vector<std::string>& extensions,
                               bool                      addEmpty);

    std::vector<std::string> _existingVideoNames;
    std::vector<std::string> _existingCalibrationNames;
    std::vector<std::string> _existingMapNames;
    std::vector<std::string> _existingVocNames;

    std::string _currentVideo;
    std::string _currentCalibration;
    std::string _currentMap;
    std::string _currentVoc;

    bool _storeKeyFrameImage;
    bool _createMarkerMap;

    WAICalibration* _wc;
};

#endif
