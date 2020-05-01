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
    SENSWebCamera();
    ~SENSWebCamera();

    //void init(SENSCameraFacing facing) override;
    void start(const Config config) override;
    void start(int width, int height) override;
    void stop() override;
    //retrieve all chamera characteristics (this may close the current capture session)
    std::vector<SENSCameraCharacteristics> getAllCameraCharacteristics() override;

    SENSFramePtr getLatestFrame() override;

private:
    void openCamera();

    bool                  _isStarting = false;
    cv::VideoCapture      _videoCapture;
    std::vector<cv::Size> _streamSizes;

    std::thread _thread;
};

//class SENSWebCameraManager : public SENSCameraManager
//{
//public:
//    SENSWebCameraManager()
//    {
//        _permissionGranted = true;
//        updateCameraCharacteristics();
//    }
//    SENSCameraPtr getOptimalCamera(SENSCameraFacing facing) override
//    {
//        //we dont know nothing anyway
//        SENSCameraPtr camera = std::unique_ptr<SENSWebCamera>(new SENSWebCamera());
//        return camera;
//    }
//
//    SENSCameraPtr getCameraForId(std::string id)
//    {
//        SENSCameraPtr camera;
//
//        auto it = _cameraInstances.find(id);
//        if (it == _cameraInstances.end())
//        {
//            for (const SENSCameraCharacteristics& c : _characteristics)
//            {
//                if (c.cameraId == id)
//                {
//                    camera               = std::shared_ptr<SENSWebCamera>(new SENSWebCamera());
//                    _cameraInstances[id] = camera;
//                }
//            }
//        }
//        else
//        {
//            camera = it->second;
//        }
//
//        return camera;
//    }
//
//protected:
//    void updateCameraCharacteristics() override
//    {
//        //There is an invisible list of devices populated from your os and your webcams appear there in the order you plugged them in.
//        //If you're e.g on a laptop with a builtin camera, that will be id 0, if you plug in an additional one, that's id 1.
//        {
//            SENSCameraCharacteristics characteristics;
//            characteristics.cameraId = "0";
//            characteristics.provided = true;
//            characteristics.streamConfig.add(cv::Size(640, 360));
//            characteristics.streamConfig.add(cv::Size(640, 480));
//            _characteristics.push_back(characteristics);
//        }
//        {
//            SENSCameraCharacteristics characteristics;
//            characteristics.cameraId = "1";
//            characteristics.provided = true;
//            characteristics.streamConfig.add(cv::Size(1920, 11080));
//            characteristics.streamConfig.add(cv::Size(640, 360));
//            characteristics.streamConfig.add(cv::Size(640, 480));
//            _characteristics.push_back(characteristics);
//        }
//    }
//};

#endif //SENS_WEBCAMERA_H
