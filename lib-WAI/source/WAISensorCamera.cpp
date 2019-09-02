#include <WAISensorCamera.h>

WAI::SensorCamera::SensorCamera(CameraCalibration* cameraCalibration)
{
    _cameraCalibration = *cameraCalibration;

    _cameraMatrix                 = cv::Mat::zeros(3, 3, CV_32F);
    _cameraMatrix.at<float>(0, 0) = cameraCalibration->fx;
    _cameraMatrix.at<float>(1, 1) = cameraCalibration->fy;
    _cameraMatrix.at<float>(0, 2) = cameraCalibration->cx;
    _cameraMatrix.at<float>(1, 2) = cameraCalibration->cy;
    _cameraMatrix.at<float>(2, 2) = 1.0f;

    _distortionMatrix                 = cv::Mat::zeros(4, 1, CV_32F);
    _distortionMatrix.at<float>(0, 0) = cameraCalibration->k1;
    _distortionMatrix.at<float>(1, 0) = cameraCalibration->k2;
    _distortionMatrix.at<float>(2, 0) = cameraCalibration->p1;
    _distortionMatrix.at<float>(3, 0) = cameraCalibration->p2;

    WAI_LOG("fx: %f, fy: %f, cx: %f, cy: %f\n",
            cameraCalibration->fx,
            cameraCalibration->fy,
            cameraCalibration->cx,
            cameraCalibration->cy);
}

void WAI::SensorCamera::update(void* cameraData)
{
    _imageGray = *((CameraData*)cameraData)->imageGray;
    _imageRGB  = *((CameraData*)cameraData)->imageRGB;

    if (_mode)
    {
        _mode->notifyUpdate();
    }
}

void WAI::SensorCamera::subscribeToUpdate(Mode* mode)
{
    _mode = mode;
}
