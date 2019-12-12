#ifndef CAM_CALIBRATION_MANAGER_H
#define CAM_CALIBRATION_MANAGER_H

#include <string>
#include <opencv2/opencv.hpp>
#include <CV/CVCalibration.h>

/*! Camera calibration manager to calculate CamCalibration
    */
class CamCalibrationManager
{
public:
    CamCalibrationManager(cv::Size boardSize  = cv::Size(16, 9),
                          cv::Size imgSize    = cv::Size(1920, 1080),
                          float    squareSize = 0.285f,
                          int      numOfImgs  = 10);
    void addCorners(const std::vector<cv::Point2f>& corners);
    bool readyForCalibration() const
    {
        return _calibCorners.size() >= _minNumImgs;
    }
    //! execute calculation of calibration using collected corners
    CVCalibration calculateCalibration(
      bool fixAspectRatio,
      bool zeroTangentDistortion,
      bool fixPrincipalPoint,
      bool calibRationalModel,
      bool calibTiltedModel,
      bool calibThinPrismModel);

    std::string getHelpMsg();
    std::string getStatusMsg();

private:
    std::vector<std::vector<cv::Point2f>> _calibCorners;
    int                                   _minNumImgs = 10;
    cv::Size                              _boardSize  = cv::Size(8, 5);
    float                                 _squareSize = 0.285f;
    cv::Size                              _imageSize  = cv::Size(1920, 1080);
};

#endif //CAM_CALIBRATION_MANAGER_H
