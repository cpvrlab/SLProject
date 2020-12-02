#ifndef SENS_IOSARCORE_H
#define SENS_IOSARCORE_H

#include <mutex>
#include <thread>
#include <atomic>

#include <sens/SENSARCore.h>

#import "SENSiOSARCoreDelegate.h"
#import <simd/simd.h>

class ImageConverter
{
public:
    //install callback to update new frame
    ImageConverter(std::function<void(SENSFrameBasePtr)> setFrameCB)
     : _setFrameCB(setFrameCB)
    {
        _converterThread = std::thread(&ImageConverter::convertAndUpdateImage, this);
    }
    
    ~ImageConverter()
    {
        terminate();
    }
    
    //convert imaga and update
    void convertAndUpdateImage()
    {
        while(!_stopThread)
        {
            std::unique_lock<std::mutex> lock(_newFrameMutex);
            _condVar.wait(lock, [&]() { return _stopThread || _newFrame; });
            if(_stopThread)
                break;
            
            //take over the latest frame
            std::unique_ptr<SENSFrameBase> currFrame = std::move(_newFrame);
            lock.unlock();
            
            HighResTimer t;
            //do the convertion
            cv::cvtColor(currFrame->imgBGR, currFrame->imgBGR, cv::COLOR_YUV2BGR_NV12, 3);
            Utils::log("ImageConverter", "cvtColor %fms", t.elapsedTimeInMilliSec());
            //transfer result with callback
            SENSFrameBasePtr convertedFrame = std::move(currFrame);
            _setFrameCB(convertedFrame);
        }
    }
    
    void setFrame(std::unique_ptr<SENSFrameBase> newFrame)
    {
        {
            std::lock_guard<std::mutex> lock(_newFrameMutex);
            _newFrame = std::move(newFrame);
        }
        _condVar.notify_one();
    }
    
private:
    void terminate()
    {
        _stopThread = true;
        _condVar.notify_one();
        if(_converterThread.joinable())
            _converterThread.join();
    }
    
    std::condition_variable _condVar;
    std::mutex _newFrameMutex;
    std::unique_ptr<SENSFrameBase> _newFrame;
    std::thread _converterThread;
    std::atomic_bool _stopThread{false};
    
    std::function<void(SENSFrameBasePtr)> _setFrameCB;
};


class SENSiOSARCore : public SENSARCore
{
public:
    SENSiOSARCore();
    ~SENSiOSARCore()
    {

    }
    
    bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) override;
    bool isReady() override;
    bool resume() override;
    void reset() override;
    void pause() override;
    SENSFramePtr latestFrame() override;
    void setDisplaySize(int w, int h) override;

private:
    void onUpdate(simd_float4x4* camPose, uint8_t* yPlane, uint8_t* uvPlane, int imgWidth, int imgHeight, simd_float3x3* camMat3x3, bool isTracking);
    void onUpdateBGR(simd_float4x4* camPose, cv::Mat imgBGR, simd_float3x3* camMat3x3);
    void setFrame(SENSFrameBasePtr frame)
    {
        std::lock_guard<std::mutex> lock(_frameMutex);
        _frame = frame;
    }
    
    SENSiOSARCoreDelegate* _arcoreDelegate;
    
    std::unique_ptr<ImageConverter> _imgConverter;
};

#endif
