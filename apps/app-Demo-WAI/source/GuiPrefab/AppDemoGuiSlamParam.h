//#############################################################################
//  File:      AppDemoGuiTestBenchOpen.h
//  Author:    Luc Girod
//  Date:      September 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_SLAM_PARAM_H
#define SL_IMGUI_SLAM_PARAM_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>

#include <SLMat4.h>
#include <SLNode.h>
#include <WAICalibration.h>
#include <vector>
#include <WAIApp.h>

class WAIApp;
//-----------------------------------------------------------------------------
class AppDemoGuiSlamParam : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiSlamParam(const std::string&              name,
                        bool*                           activator,
                        std::queue<WAIEvent*>*          eventQueue,
                        const std::vector<std::string>& extractorIdToNames);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

private:
    int _currentId;
    int _iniCurrentId;
    int _markerCurrentId;

    const std::vector<std::string>& _extractorIdToNames;
    std::queue<WAIEvent*>*          _eventQueue;
};

#endif
