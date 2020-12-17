#ifndef VIDEO_BACKGROUND_CAMERA_H
#define VIDEO_BACKGROUND_CAMERA_H

#include <string>
#include <SLCamera.h>
#include <SLGLTexture.h>

class VideoBackgroundCamera : public SLCamera
{
public:
    VideoBackgroundCamera(std::string cameraName, std::string defaultBackgroundImage, std::string shaderPath)
      : SLCamera(cameraName)
    {
        _videoTexture = new SLGLTexture(nullptr, defaultBackgroundImage, GL_LINEAR, GL_LINEAR);
        _background.texture(_videoTexture, false);

        // Define shader that shows on all pixels the video background
        _spVideoBackground  = new SLGLProgramGeneric(nullptr,
                                                    shaderPath + "PerPixTmBackground.vert",
                                                    shaderPath + "PerPixTmBackground.frag");
        _matVideoBackground = new SLMaterial(nullptr,
                                             "matVideoBackground",
                                             _videoTexture,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             _spVideoBackground);
    }

    ~VideoBackgroundCamera()
    {
        delete _videoTexture;
        delete _spVideoBackground;
        delete _matVideoBackground;
    }

    void updateVideoImage(const cv::Mat& image)
    {
        _videoTexture->copyVideoImage(image.cols,
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

    SLMaterial* matVideoBackground() { return _matVideoBackground; }

protected:
    SLGLTexture* _videoTexture       = nullptr;
    SLGLProgram* _spVideoBackground  = nullptr;
    SLMaterial*  _matVideoBackground = nullptr;
};

#endif
