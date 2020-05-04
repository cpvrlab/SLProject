#include "SENSVideoStream.h"
#include "SENS.h"
#include "SENSException.h"
#include <Utils.h>
#include "SENSUtils.h"

SENSVideoStream::SENSVideoStream(const std::string& videoFileName,
                                 bool videoLoops, 
                                 bool mirrorH, 
                                 bool mirrorV, 
                                 float targetFps)
  : _videoLoops(videoLoops),
    _mirrorH(mirrorH),
    _mirrorV(mirrorV)
{
    if (!Utils::fileExists(videoFileName))
    {
        throw SENSException(SENSType::VIDEO, "Video file does not exist: " + videoFileName, __LINE__, __FILE__);
    }

    _cap.open(videoFileName);
    if (!_cap.isOpened())
        throw SENSException(SENSType::VIDEO, "Could not open video file stream: " + videoFileName, __LINE__, __FILE__);

    _videoFrameSize = {(int)_cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)_cap.get(cv::CAP_PROP_FRAME_HEIGHT)};
    _frameCount     = (int)_cap.get(cv::CAP_PROP_FRAME_COUNT);
    _fps            = (float)_cap.get(cv::CAP_PROP_FPS);

    if (targetFps == 0)
        _targetFps = _fps;
    else
        _targetFps = targetFps;


    _videoFileName  = videoFileName;
}

SENSFramePtr SENSVideoStream::grabNextFrame()
{
    if (!_cap.isOpened())
        throw SENSException(SENSType::VIDEO, "Video file stream is not open!", __LINE__, __FILE__);

    if (_videoLoops)
    {
        if (_cap.get(cv::CAP_PROP_POS_FRAMES) == _frameCount)
            _cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    SENSFramePtr sensFrame;
    cv::Mat      rgbImg;
    cv::Mat      grayImg;

    if (_cap.read(rgbImg))
    {
        SENS::mirrorImage(rgbImg, _mirrorH, _mirrorV);
        cv::cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);

        sensFrame = std::make_shared<SENSFrame>(
          rgbImg,
          grayImg,
          rgbImg.size().width,
          rgbImg.size().height,
          0,
          0,
          _mirrorH,
          _mirrorV);
    }

    return sensFrame;
}

SENSFramePtr SENSVideoStream::grabNextResampledFrame()
{
    if (!_cap.isOpened())
        throw SENSException(SENSType::VIDEO, "Video file stream is not open!", __LINE__, __FILE__);

    if (_videoLoops)
    {
        if (_cap.get(cv::CAP_PROP_POS_FRAMES) == _frameCount)
            _cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    SENSFramePtr sensFrame;
    cv::Mat      rgbImg;
    cv::Mat      grayImg;

    float frameDuration = 1.0f / _targetFps;
    int frameIndex = (int)_cap.get(cv::CAP_PROP_POS_FRAMES);
    int skippedFrame = 0;
    float frameTime;
    float lastFrameTime = frameIndex / _fps;
    do
    {
        skippedFrame++;
        frameTime = (float)(frameIndex + skippedFrame) / _fps;
    }
    while ((frameTime - lastFrameTime) < frameDuration);

    float frameError = (frameTime - lastFrameTime - frameDuration);  //Compute overflow
    int nbFrame = (int)(frameIndex * _targetFps / _fps);             //Expected number of frames
    float totalError = (nbFrame * frameError);                       //Total cumulated error
    if ((fmod(totalError, frameDuration) + frameError) > frameDuration && skippedFrame > 1)
    {
        skippedFrame--;
    }

    moveCapturePosition(skippedFrame);

    if (_cap.read(rgbImg))
    {
        SENS::mirrorImage(rgbImg, _mirrorH, _mirrorV);
        cv::cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);

        sensFrame = std::make_shared<SENSFrame>(
          rgbImg,
          grayImg,
          rgbImg.size().width,
          rgbImg.size().height,
          0,
          0,
          _mirrorH,
          _mirrorV);
    }

    return sensFrame;
}

SENSFramePtr SENSVideoStream::grabPreviousResampledFrame()
{
    int frameIndex = (int)_cap.get(cv::CAP_PROP_POS_FRAMES);
    int skippedFrame = 0;
    float frameTime;
    float currentFrameTime = frameIndex / _fps;

    do
    {
        frameTime = frameIndex - skippedFrame / _fps;
        skippedFrame++;
    }
    while ((currentFrameTime - frameTime) > 1.0 / _targetFps);

    moveCapturePosition(-skippedFrame+1);
    return grabNextFrame();
}

SENSFramePtr SENSVideoStream::grabPreviousFrame()
{
    moveCapturePosition(-2);
    return grabNextFrame();
}

void SENSVideoStream::moveCapturePosition(int n)
{
    int frameIndex = (int)_cap.get(cv::CAP_PROP_POS_FRAMES);
    frameIndex += n;

    if (frameIndex < 0)
        frameIndex = 0;
    else if (frameIndex > _frameCount)
        frameIndex = _frameCount;

    _cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
}
