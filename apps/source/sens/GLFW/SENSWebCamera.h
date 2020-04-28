#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <opencv2/opencv.hpp>
#include <SENSCamera.h>
#include <thread>

class SENSWebCameraManager;

class SENSWebCamera : public SENSCamera
{
    friend class SENSWebCameraManager;
    friend class SENSCameraManager;

public:
    ~SENSWebCamera();

    //void init(SENSCameraFacing facing) override;
    void start(const Config config) override;
    void start(int width, int height) override;
    void stop() override;

    SENSFramePtr getLatestFrame() override;

private:
    SENSWebCamera();

    void openCamera();

    bool                  _isStarting = false;
    cv::VideoCapture      _videoCapture;
    std::vector<cv::Size> _streamSizes;

    std::thread _thread;
};

class SENSWebCameraManager : public SENSCameraManager
{
public:
    SENSWebCameraManager()
    {
        updateCameraCharacteristics();
    }
    SENSCameraPtr getOptimalCamera(SENSCameraFacing facing) override
    {
        //we dont know nothing anyway
        SENSCameraPtr camera = std::unique_ptr<SENSWebCamera>(new SENSWebCamera());
        return std::move(camera);
    }

    SENSCameraPtr getCameraForId(std::string id)
    {
        SENSCameraPtr camera;
        for (const SENSCameraCharacteristics& c : _characteristics)
        {
            if (c.cameraId == id)
            {
                camera = std::unique_ptr<SENSWebCamera>(new SENSWebCamera());
            }
        }
        return std::move(camera);
    }

protected:
    void updateCameraCharacteristics() override
    {
        //There is an invisible list of devices populated from your os and your webcams appear there in the order you plugged them in.
        //If you're e.g on a laptop with a builtin camera, that will be id 0, if you plug in an additional one, that's id 1.
        SENSCameraCharacteristics characteristics;
        characteristics.cameraId = "0";
        characteristics.provided = false;

        _characteristics.push_back(characteristics);
    }
};

#endif //SENS_WEBCAMERA_H
