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
    virtual bool init() = 0;
    virtual bool isReady()                                                                                         = 0;
    virtual bool resume()                                                                                          = 0;
    virtual void reset()                                                                                           = 0;
    virtual void pause()                                                                                           = 0;
    //! Returns true if in tracking state. If correctly initialized, it will update the camera frame that may be retrieved with latestFrame()
    virtual bool update(cv::Mat& view) { return false; }
    virtual void lightComponentIntensity(float * components) { }

    bool         isAvailable() { return _available; };
    bool         isRunning() { return !_pause; }

protected:
    bool             _running = false;

    bool _available = false;
    bool _pause     = true;

    int _inputFrameW = 0;
    int _inputFrameH = 0;
private:
};

#endif
