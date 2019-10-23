//#############################################################################
//  File:      AppDemoGuiVideoStorage.h
//  Author:    Luc Girod
//  Date:      April 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_IMGUI_VIDEOSTORAGE_H
#define SL_IMGUI_VIDEOSTORAGE_H

#include <opencv2/core.hpp>
#include <AppDemoGuiInfosDialog.h>

#include <SLMat4.h>
#include <SLNode.h>

//-----------------------------------------------------------------------------
class AppDemoGuiVideoStorage : public AppDemoGuiInfosDialog
{
    public:
    AppDemoGuiVideoStorage(const std::string& name,
                           cv::VideoWriter*   videoWriter,
                           cv::VideoWriter*   videoWriterInfo,
                           std::ofstream*     gpsDataStream,
                           bool*              activator);

    void buildInfos(SLScene* s, SLSceneView* sv) override;

    private:
    void saveVideo(std::string filename);
    void saveGPSData(std::string videofile);

    ofstream*        _gpsDataFile;
    cv::VideoWriter* _videoWriter;
    cv::VideoWriter* _videoWriterInfo;
};

#endif //SL_IMGUI_VIDEOSTORAGE_H
