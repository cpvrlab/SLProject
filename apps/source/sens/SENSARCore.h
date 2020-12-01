#ifndef SENS_ARCORE_H
#define SENS_ARCORE_H

#include <mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <Utils.h>
#include <opencv2/opencv.hpp>
#include "SENS.h"

#include <sens/SENSFrame.h>
#include <sens/SENSCamera.h>

class SENSARCore //: public SENSCameraBase
{
public: 

    struct config
    {
        //! largest target image width (only BGR)
        int targetWidth;
        //! largest target image width (only BGR)
        int targetHeight;
        //! width of smaller image version (e.g. for tracking)
        int manipWidth;
        //! height of smaller image version (e.g. for tracking)
        int manipHeight;
        //! provide gray version of small image
        bool convertManipToGray;
    };

    SENSARCore() {}
    virtual ~SENSARCore() {}
    virtual bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) = 0;
    virtual bool isReady() = 0;
    virtual bool resume() = 0;
    virtual void reset() = 0;
    virtual void pause() = 0;
    virtual bool update(cv::Mat& intrinsic, cv::Mat& view) { return false; }
    virtual SENSFramePtr latestFrame() = 0;
    virtual void setDisplaySize(int w, int h) = 0;

    bool isAvailable() { return _available; };
    bool isRunning() { return !_pause; }

protected:
    SENSFramePtr processNewFrame(const SENSTimePt& timePt, cv::Mat& bgrImg, cv::Mat intrinsics);

    void configure(int  targetWidth,
                           int  targetHeight,
                           int  manipWidth,
                           int  manipHeight,
                           bool convertManipToGray);
    
    bool             _running = false;
    std::mutex       _frameMutex;
    SENSFrameBasePtr _frame;
    struct config    _config;

    bool _available = false;
    bool _pause     = true;
private:
};

#endif
