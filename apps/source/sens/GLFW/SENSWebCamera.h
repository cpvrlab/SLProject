#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <opencv2/opencv.hpp>
#include <sens/SENSCamera.h>

class SENSWebCamera : public SENSCameraBase
{
public:
    void start(const SENSCameraConfig config) override;
    void start(std::string id, int width, int height) override;
    void stop() override;
    //retrieve all chamera characteristics (this may close the current capture session)
    std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics() override;

    SENSFramePtr getLatestFrame() override;

private:
    cv::VideoCapture _videoCapture;
};

#endif //SENS_WEBCAMERA_H
