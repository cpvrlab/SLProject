#ifndef VIDEO_BACKGROUND_CAMERA_H
#define VIDEO_BACKGROUND_CAMERA_H

#include <string>
#include <SLCamera.h>
#include <SLGLTexture.h>

class VideoBackgroundCamera : public SLCamera
{
public:
    VideoBackgroundCamera(std::string cameraName, std::string defaultBackgroundImage)
     : SLCamera(cameraName)
    {
        _videoImage = new SLGLTexture(nullptr, defaultBackgroundImage, GL_LINEAR, GL_LINEAR);
        this->background().texture(_videoImage, false);
    }
    
    ~VideoBackgroundCamera()
    {
        delete _videoImage;
    }
    
    void updateVideoImage(const cv::Mat& image)
    {
        _videoImage->copyVideoImage(image.cols,
                                    image.rows,
                                    CVImage::cv2glPixelFormat(image.type()),
                                    image.data,
                                    image.isContinuous(),
                                    true);
    }
    
    void updateCameraIntrinsics(float cameraFovVDeg)
    {
        this->fov(cameraFovVDeg);
        // Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
        //std::cout << "cameraMatUndistorted: " << cameraMatUndistorted << std::endl;
        /*
        cameraNode->intrinsics((float)cameraMatUndistorted.at<double>(0, 0),
                               (float)cameraMatUndistorted.at<double>(1, 1),
                               (float)cameraMatUndistorted.at<double>(0, 2),
                               (float)cameraMatUndistorted.at<double>(1, 2));
        */
        //enable projection -> intrinsics mode
        //cameraNode->projection(P_monoIntrinsic);
        this->projection(P_monoPerspective);
    }
    
protected:

    SLGLTexture* _videoImage = nullptr;
};

#endif
