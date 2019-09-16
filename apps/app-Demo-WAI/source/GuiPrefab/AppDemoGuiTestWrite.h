//#############################################################################
//  File:      AppDemoGuiMapStorage.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_TEST_WRITE_H
#define SL_IMGUI_TEST_WRITE_H

#include <string>
#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>

#include <SLMat4.h>
#include <SLNode.h>
#include <WAICalibration.h>
#include <WAI.h>

//-----------------------------------------------------------------------------
class AppDemoGuiTestWrite : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiTestWrite(const std::string& name, std::string saveDir,
                        WAI::WAI* wai, WAICalibration* wc, SLNode* mapNode,
                        cv::VideoWriter* writer1, cv::VideoWriter* writer2,
                        std::ofstream* gpsDataStream,
                        bool* activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:

    void prepareExperiment(std::string testScene, std::string weather);

    void recordExperiment();
    void stopRecording();

    void saveGPSData(std::string videofile);
    void saveRunVideo(std::string run);
    void saveVideo(std::string video);
    void saveCalibration(std::string calib);
    void saveMap(std::string map);
    void saveTestSettings(std::string path);

    std::string _baseDir;
    std::string _mapDir;

    std::string _savePath;
    std::string _settingsPath;
    std::string _videoPath;
    std::string _mapPath;
    std::string _runPath;
    std::string _date;

    std::ofstream*           _gpsDataFile;
    SLNode*                  _mapNode;
    std::vector<std::string> _testScenes;
    std::vector<std::string> _conditions;
    cv::VideoWriter*         _videoWriter;
    cv::VideoWriter*         _videoWriterInfo;
    WAICalibration*          _wc;
    WAI::WAI*                _wai;

    int _currentSceneId;
    int _currentConditionId;
};

#endif
