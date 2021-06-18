#ifndef SENS_ARCORE_H
#define SENS_ARCORE_H

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

class SENSARCore : public SENSCameraBase
{
public:
    SENSARCore() {}
    virtual ~SENSARCore() {}
    virtual bool init(unsigned int textureId=0, bool retrieveCpuImg=false, int targetWidth=-1, int targetHeight=-1) = 0;
    virtual bool isReady()                                                                                         = 0;
    virtual bool resume()                                                                                          = 0;
    virtual void reset()                                                                                           = 0;
    virtual void pause()                                                                                           = 0;
    //! Returns true if in tracking state. If correctly initialized, it will update the camera frame that may be retrieved with latestFrame()
    virtual bool update(cv::Mat& view) { return false; }
    virtual void lightComponentIntensity(float * components) { }

    virtual bool isInstalled() { return false; };
    virtual bool isAvailable() { return false; };
    virtual bool install() { return false; };
    virtual bool installRefused() { return false; };
    virtual void installRefused(bool b) { };
    bool         isRunning() { return !_pause; }
    
    void fetchPointCloud(bool s) { _fetchPointCloud = s; }
    cv::Mat getPointCloud() { return _pointCloud; }

protected:
    bool _running   = false;
    bool _pause     = true;

    int _inputFrameW = 0;
    int _inputFrameH = 0;

    bool _fetchPointCloud = false;
    cv::Mat _pointCloud;
private:
};

#endif
