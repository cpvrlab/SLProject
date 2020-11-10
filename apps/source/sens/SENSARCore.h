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
class SENSARCore
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
    virtual bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) { return false; }
    virtual bool isReady() { return false; }
    virtual bool resume() { return true; }
    virtual void pause() {}
    virtual bool update(cv::Mat& intrinsic, cv::Mat& view) { return false; }
    virtual SENSFramePtr latestFrame() { return nullptr; }
    virtual void setDisplaySize(int w, int h) { return; }

    virtual void configure(int  targetWidth,
                           int  targetHeight,
                           int  manipWidth,
                           int  manipHeight,
                           bool convertManipToGray);

protected:
    bool             _running = false;
    SENSFrameBasePtr _frame;
    struct config    _config;

private:
};

#endif
