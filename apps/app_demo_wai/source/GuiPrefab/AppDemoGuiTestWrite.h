//#############################################################################
//  File:      AppDemoGuiTestWrite.h
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

#include <CVCalibration.h>

class WAIApp;

//-----------------------------------------------------------------------------
class AppDemoGuiTestWrite : public AppDemoGuiInfosDialog
{
public:
    AppDemoGuiTestWrite(const std::string& name,
                        CVCalibration*     calib,
                        SLNode*            mapNode,
                        cv::VideoWriter*   writer1,
                        cv::VideoWriter*   writer2,
                        std::ofstream*     gpsDataStream,
                        bool*              activator,
                        WAIApp&            waiApp);

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

    std::string _date;
    std::string gpsname;
    std::string videoname;
    std::string runvideoname;
    std::string calibrationname;
    std::string settingname;
    std::string mapname;

    cv::Size                 _size;
    std::ofstream*           _gpsDataFile;
    SLNode*                  _mapNode;
    std::vector<std::string> _testScenes;
    std::vector<std::string> _conditions;
    cv::VideoWriter*         _videoWriter;
    cv::VideoWriter*         _videoWriterInfo;
    CVCalibration*           _calib;

    int _currentSceneId;
    int _currentConditionId;

    WAIApp& _waiApp;
};

#endif
