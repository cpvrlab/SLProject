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
    struct Config
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
    virtual bool isReady()                                                                                         = 0;
    virtual bool resume()                                                                                          = 0;
    virtual void reset()                                                                                           = 0;
    virtual void pause()                                                                                           = 0;
    //! Returns true if in tracking state. If correctly initialized, it will update the camera frame that may be retrieved with latestFrame()
    virtual bool update(cv::Mat& view) { return false; }

    //! Get the latest camera frame. You have to call update() first to get a new frame.
    SENSFramePtr latestFrame();
    bool         isAvailable() { return _available; };
    bool         isRunning() { return !_pause; }

    const SENSCalibration* const calibration() const { return _calibration.get(); }
    const SENSCalibration* const calibrationManip() const { return _calibrationManip.get(); }

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
    Config           _config;

    bool _available = false;
    bool _pause     = true;

    int _inputFrameW = 0;
    int _inputFrameH = 0;
    //! The calibration is used for computer vision applications. This calibration is adjusted to fit to the original sized image (see SENSFrame::imgBGR and SENSCameraConfig::targetWidth, targetHeight)
    std::unique_ptr<SENSCalibration> _calibration;
    //calibration that fits to (targetWidth,targetHeight)
    std::unique_ptr<SENSCalibration> _calibrationManip;

private:
};

#endif
