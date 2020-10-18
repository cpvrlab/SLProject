#include "SENSCvCamera.h"
#include <opencv2/imgproc.hpp>
#include <sens/SENSUtils.h>
#include <Utils.h>

//we scale the original image only to meet the width of manipImg. After that the targetImg may be additionally cropped (e.g. to screen width)
SENSFramePtr SENSCvCamera::postProcess(cv::Mat& bgrImg, cv::Mat intrinsics, bool intrinsicsChanged)
{
    cv::Size inputSize = bgrImg.size();
    // Mirroring (we dont need it at the moment)
    SENS::mirrorImage(bgrImg, _config->mirrorH, _config->mirrorV);

    //scale original bgrImg to manipulated image. We dont crop the manipulated image
    cv::Mat manipImg;
    float   scale = 1.0f;
    if (_config->manipWidth > 0 && _config->manipHeight > 0)
    {
        int cropW = 0, cropH = 0;
        SENS::cropImageTo(bgrImg, manipImg, (float)_config->manipWidth / (float)_config->manipHeight, cropW, cropH);
        scale = (float)_config->manipWidth / (float)manipImg.size().width;
        cv::resize(bgrImg, manipImg, cv::Size(), scale, scale);
    }
    else if (_config->convertManipToGray)
    {
        manipImg = bgrImg;
    }

    // Create grayscale
    if (_config->convertManipToGray)
        cv::cvtColor(manipImg, manipImg, cv::COLOR_BGR2GRAY);

    // Crop Video image to required aspect ratio
    if (_config->targetWidth != bgrImg.cols &&
        _config->targetHeight != bgrImg.rows)
    {
        int cropW = 0, cropH = 0;
        SENS::cropImage(bgrImg, (float)_config->targetWidth / (float)_config->targetHeight, cropW, cropH);
    }

    SENSFramePtr sensFrame = std::make_unique<SENSFrame>(bgrImg,
                                                         manipImg,
                                                         _config->mirrorH,
                                                         _config->mirrorV,
                                                         1 / scale,
                                                         intrinsics);

    return sensFrame;
}

//TODO: we scale the original image only to meet the width of manipImg. After that the targetImg may be additionally cropped (e.g. to screen width)
SENSFramePtr SENSCvCamera::processNewFrame(cv::Mat& bgrImg, cv::Mat intrinsics, bool intrinsicsChanged)
{
    //todo: accessing config readonly should be no problem  here, as the config only changes when camera is stopped
    cv::Size inputSize = bgrImg.size();

    // Crop Video image to required aspect ratio
    int cropW = 0, cropH = 0;
    SENS::cropImage(bgrImg, (float)_config->targetWidth / (float)_config->targetHeight, cropW, cropH);

    // Mirroring
    SENS::mirrorImage(bgrImg, _config->mirrorH, _config->mirrorV);

    cv::Mat manipImg;
    float   scale = 1.0f;
    //problem: eingangsbild 16:9 -> targetImg 4:3 -> crop left and right -> manipImg 16:9 -> weiterer crop oben und unten -> FALSCH
    if (_config->manipWidth > 0 && _config->manipHeight > 0)
    {
        manipImg  = bgrImg;
        int cropW = 0, cropH = 0;
        SENS::cropImage(manipImg, (float)_config->manipWidth / (float)_config->manipHeight, cropW, cropH);
        scale = (float)_config->manipWidth / (float)manipImg.size().width;
        cv::resize(manipImg, manipImg, cv::Size(), scale, scale);
    }
    else if (_config->convertManipToGray)
    {
        manipImg = bgrImg;
    }

    // Create grayscale
    if (_config->convertManipToGray)
    {
        cv::cvtColor(manipImg, manipImg, cv::COLOR_BGR2GRAY);
    }

    SENSFramePtr sensFrame = std::make_unique<SENSFrame>(bgrImg,
                                                         manipImg,
                                                         _config->mirrorH,
                                                         _config->mirrorV,
                                                         1 / scale,
                                                         intrinsics);

    return sensFrame;
}

void SENSCvCamera::guessAndSetCalibration(float fovDegFallbackGuess)
{
    //We make a calibration with full resolution and adjust it to the manipulated image size later if neccessary:
    //For the initial setup we have to use streamconfig values because that is where the fov fits too
    float horizFOVDev = fovDegFallbackGuess;
    if (_config->streamConfig->focalLengthPix > 0)
        horizFOVDev = SENS::calcFOVDegFromFocalLengthPix(_config->streamConfig->focalLengthPix, _config->streamConfig->widthPix);

    //init calibration with streaconfig image size, because that is where the fov fits too
    _calibration = std::make_unique<SENSCalibration>(cv::Size(_config->streamConfig->widthPix, _config->streamConfig->heightPix),
                                                     horizFOVDev,
                                                     false,
                                                     false,
                                                     SENSCameraType::BACKFACING, //todo
                                                     Utils::ComputerInfos().get());
    //now we adapt the calibration to the target size
    if (_config->targetWidth != _config->streamConfig->widthPix || _config->targetHeight != _config->streamConfig->heightPix)
        _calibration->adaptForNewResolution({_config->targetWidth, _config->targetHeight}, false);

    //update second calibration
    _calibrationManip = std::make_unique<SENSCalibration>(*_calibration);
    _calibrationManip->adaptForNewResolution(cv::Size(_config->manipWidth, _config->manipHeight), false);
}

SENSCvCamera::SENSCvCamera(SENSCamera* camera)
  : _camera(camera)
{
    assert(camera);
    //retrieve capture properties
    _camera->captureProperties();
}

bool SENSCvCamera::supportsFacing(SENSCameraFacing facing)
{
    const SENSCaptureProperties& props = _camera->captureProperties();
    return props.supportsCameraFacing(facing);
}

//todo:
//configure to reach maximum field of view
/*
 we are interestest in maxmimum vertical field of view for a perspective camera (that can be calibrated with pinhole camera model).
 This is interesting for tracking and ar visualization on a small display.
 we can still crop the image to the screen if we like
ConfigReturnCode configureForMaxFoc();
 */

/*!
 configure camera: the function checks with the wrapped camera if it could successfully configure the camera.
 If not, it will return false. This may have dirrerent reasons:
 - the wrapped camera is not valid
 - it has to extrapolate the maximum found stream config size to get targetWidth, targetHeight.
 Transfer -1 for manipWidth and manipHeight to disable generation of manipulated image
 */
//current limitation: aspect ratios are the same for target and manip img:
//targetW / targetH defines aspect ratio.
SENSCvCamera::ConfigReturnCode SENSCvCamera::configure(SENSCameraFacing facing,
                                                       int              targetWidth,
                                                       int              targetHeight,
                                                       int              manipWidth,
                                                       int              manipHeight,
                                                       bool             mirrorH,
                                                       bool             mirrorV,
                                                       bool             convertManipToGray)
{
    //this can happen, because it is possible to retrieve a camera pointer reference for camera simulation
    if (!_camera)
        return ERROR_CAMERA_INVALID;

    if (_camera->started())
        return ERROR_CAMERA_IS_RUNNING;

    const SENSCaptureProperties& props = _camera->captureProperties();
    //check if facing is available
    if (!props.supportsCameraFacing(facing))
        return ERROR_FACING_NOT_AVAILABLE;

    ConfigReturnCode returnCode = SUCCESS;

    //search for best matching stream config
    int   trackingImgW = 640;
    float searchWdivH;
    if (((float)targetWidth / (float)targetHeight) >
        ((float)manipWidth / (float)manipHeight))
        searchWdivH = (float)manipWidth / (float)manipHeight;
    else
        searchWdivH = (float)targetWidth / (float)targetHeight;

    //approximately what resolution we search for visualiation image
    int aproxHighImgW = targetWidth;

    std::pair<const SENSCameraDeviceProperties*, const SENSCameraStreamConfig*> bestConfig =
      props.findBestMatchingConfig(facing, 65.f, aproxHighImgW, (int)((float)aproxHighImgW / searchWdivH));

    //warn if extrapolation needed (image will not be extrapolated)
    if (!bestConfig.first && !bestConfig.second)
    {
        //we reduce this size
        aproxHighImgW = 640;
        bestConfig    = props.findBestMatchingConfig(facing, 65.f, aproxHighImgW, (int)((float)aproxHighImgW / searchWdivH));
        if (!bestConfig.first && !bestConfig.second)
            return ERROR_UNKNOWN;
        else
            returnCode = WARN_TARGET_SIZE_NOT_FOUND;
    }

    _config = std::make_unique<SENSCvCameraConfig>(bestConfig.first,
                                                   bestConfig.second,
                                                   targetWidth,
                                                   targetHeight,
                                                   manipWidth,
                                                   manipHeight,
                                                   mirrorH,
                                                   mirrorV,
                                                   convertManipToGray);

    //guess calibrations if no calibration is set from outside
    if(!_calibrationOverwrite)
        guessAndSetCalibration(65.f);

    return returnCode;
}

bool SENSCvCamera::start()
{
    if (!_camera)
        return false;

    if (!_config)
        return false;

    _camera->start(_config->deviceProps->deviceId(),
                   *_config->streamConfig,
                   true);

    return true;
}

bool SENSCvCamera::started()
{
    if (_camera)
        return _camera->started();
    else
        return false;
}

void SENSCvCamera::stop()
{
    if (_camera)
        _camera->stop();
}

SENSFramePtr SENSCvCamera::latestFrame()
{
    SENSFramePtr latestFrame;

    SENSFrameBasePtr frameBase = _camera->latestFrame();
    if (frameBase)
    {
        //process
        latestFrame = processNewFrame(frameBase->imgBGR, frameBase->intrinsics, !frameBase->intrinsics.empty());
    }

    //update calibration if necessary
    //we wont update it, if overwrite calibration is valid (set from outside)
    if (latestFrame && !_calibrationOverwrite && !latestFrame->intrinsics.empty())
    {
        HighResTimer t;
        //todo: mutex for calibration?
        _calibration = std::make_unique<SENSCalibration>(latestFrame->intrinsics,
                                                         cv::Size(_config->streamConfig->widthPix, _config->streamConfig->heightPix),
                                                         _calibration->isMirroredH(),
                                                         _calibration->isMirroredV(),
                                                         _calibration->camType(),
                                                         _calibration->computerInfos());
        //now we adapt the calibration to the target size
        if (_config->targetWidth != _config->streamConfig->widthPix || _config->targetHeight != _config->streamConfig->heightPix)
            _calibration->adaptForNewResolution({_config->targetWidth, _config->targetHeight}, false);

        //update second calibration
        _calibrationManip = std::make_unique<SENSCalibration>(*_calibration);
        _calibrationManip->adaptForNewResolution(cv::Size(_config->manipWidth, _config->manipHeight), false);

        SENS_DEBUG("calib update duration %f", t.elapsedTimeInMilliSec());
    }

    return latestFrame;
}

cv::Mat SENSCvCamera::scaledCameraMat()
{
    if (!_calibrationManip)
        return cv::Mat();
    else
        return _calibrationManip->cameraMat();
}

//! Set calibration and adapt it to current image size. Camera has to be started, before this function is called.
void SENSCvCamera::setCalibration(const SENSCalibration& calibration, bool buildUndistortionMaps)
{
    _calibrationOverwrite = std::make_unique<SENSCalibration>(calibration);
    
    //update first calibration
    _calibration = std::make_unique<SENSCalibration>(*_calibrationOverwrite);
    _calibration->adaptForNewResolution(cv::Size(_config->targetWidth, _config->targetHeight), buildUndistortionMaps);
    
    //update second calibration
    _calibrationManip = std::make_unique<SENSCalibration>(*_calibrationOverwrite);
    _calibrationManip->adaptForNewResolution(cv::Size(_config->manipWidth, _config->manipHeight), buildUndistortionMaps);
}
