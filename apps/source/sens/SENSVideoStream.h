#ifndef SENS_VIDEOSTREAM_H
#define SENS_VIDEOSTREAM_H

#include <opencv2/opencv.hpp>
#include "SENSFrame.h"

class SENSVideoStream
{
public:
    SENSVideoStream(std::string videoFileName, bool videoLoops, bool mirrorH, bool mirrorV);
    SENSFramePtr grabNextFrame();
    SENSFramePtr grabPreviousFrame();

    cv::Size getFrameSize() const { return _videoFrameSize; }

    const std::string& videoFilename() const { return _videoFileName; }
    int                nextFrameIndex() const { return (int)_cap.get(cv::CAP_PROP_POS_FRAMES); }
    int                frameCount() const { return _frameCount; }
    float              fps() const { return _fps; }

private:
    void moveCapturePosition(int n);

    cv::VideoCapture _cap;
    cv::Size2i       _videoFrameSize;
    int              _frameCount = 0;
    std::string      _videoFileName;

    bool  _videoLoops = false;
    float _fps        = 0.f;

    bool _mirrorH = false;
    bool _mirrorV = false;
};

#endif //SENS_VIDEOSTREAM_H
