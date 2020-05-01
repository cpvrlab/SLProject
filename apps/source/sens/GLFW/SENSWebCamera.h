#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <opencv2/opencv.hpp>
#include <SENSCamera.h>

class SENSWebCamera : public SENSCamera
{
public:
    void start(const Config config) override;
    void start(std::string id, int width, int height) override;
    void stop() override;
    //retrieve all chamera characteristics (this may close the current capture session)
    std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics() override;

    SENSFramePtr getLatestFrame() override;

private:
    cv::VideoCapture _videoCapture;
};

#endif //SENS_WEBCAMERA_H
