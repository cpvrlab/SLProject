//#############################################################################
//  File:      SENSARBaseCamera.h
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  License:   This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SENS_ARBASECAMERA_H
#define SENS_ARBASECAMERA_H

#include <mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <Utils.h>
#include <opencv2/opencv.hpp>
#include "SENS.h"

#include <SENSFrame.h>
#include <SENSCalibration.h>
#include <SENSCamera.h>

//-----------------------------------------------------------------------------
/// Provides interface for the AR frameworks such as ARKit and ARCore
class SENSARBaseCamera : public SENSBaseCamera
{
public:
    SENSARBaseCamera() {}
    virtual ~SENSARBaseCamera() {}
    virtual bool init(unsigned int textureId      = 0,
                      bool         retrieveCpuImg = false,
                      int          targetWidth    = -1) = 0;
    virtual bool isReady()                  = 0;
    virtual bool resume()                   = 0;
    virtual void reset()                    = 0;
    virtual void pause()                    = 0;

    /*! Returns true if in tracking state. If correctly initialized, it will
    update the camera frame that may be retrieved with latestFrame() */
    virtual bool update(cv::Mat& view) { return false; }
    virtual void lightComponentIntensity(float* components) {}

    virtual bool isInstalled() { return false; };
    virtual bool isAvailable() { return false; };
    virtual bool install() { return false; };
    virtual bool installRefused() { return false; };
    virtual void installRefused(bool b){};
    bool         isRunning() { return !_pause; }

    void     fetchPointCloud(bool s) { _fetchPointCloud = s; }
    cv::Mat  pointCloud() { return _pointCloud; }
    cv::Size inputFrameSize() { return cv::Size(_inputFrameW, _inputFrameH); }
    cv::Size cpuFrameSize() { return cv::Size(_cpuImgTargetWidth, _cpuImgTargetHeight); }

protected:
    bool    _running            = false;
    bool    _pause              = true;
    int     _inputFrameW        = 0;
    int     _inputFrameH        = 0;
    bool    _retrieveCpuImg     = false;
    int     _cpuImgTargetWidth  = -1;
    int     _cpuImgTargetHeight = -1;
    bool    _fetchPointCloud    = false;
    cv::Mat _pointCloud;
};
//-----------------------------------------------------------------------------
#endif
