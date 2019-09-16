#ifndef CAM_CALIBRATION_H
#define CAM_CALIBRATION_H

#include <string>
#include <opencv2/opencv.hpp>

class CamCalibrationManager;

/*! \brief Camera calibration for perspective camera model.
     */
class CamCalibration
{
    friend CamCalibrationManager;

    public:
    //! loads a calibration file from given path
    CamCalibration(std::string filePath);
    /*! defines a calibration estimation containing only intrinsics without distortion
            \param imgSize size of image the calibration is used for
            \param horizFOV horizontal field of view in degree
         */
    CamCalibration(cv::Size imgSize, double horizFOV = 65);
    /*! Defines a calibration estimation containing only intrinsics without distortion.
            One may only use this kind of calibration if the image uses the whole sensor width (no cropping left and right)
            \param imgSize size of image the calibration is used for
            \param camFocalLengthMM camera sensor size in mm
            \param camSensorSizeMM camera focal length in mm
         */
    CamCalibration(cv::Size imgSize, float camFocalLengthMM, cv::Size2f camSensorSizeMM);
    //! copy constructor
    CamCalibration(const CamCalibration& toCopy);

    //!getters
    const cv::Mat& getCameraMat() const
    {
        return _cameraMat;
    }
    const cv::Mat& getDistortion() const
    {
        return _distortion;
    }
    const cv::Size& getImgSize() const
    {
        return _imgSize;
    }
    const float getHorizontalFOV() const;

    //! scale calibration after changing image size
    void scale(double scaleFactor);
    //! save calibration to given file path
    void save(const std::string& filePath);

    private:
    //! make default ctor private so no empty CamCalibration is possible.
    CamCalibration(){};
    //! load camera calibration from given path
    void load(std::string filePath);

    //! 3x3 Matrix for intrinsic camera matrix
    cv::Mat _cameraMat;
    //! 4x1 Matrix for intrinsic distortion
    cv::Mat _distortion;

    //! day time when calibration was calculated
    std::string _calibrationTime;
    //! reprojection error
    double _reprojectionError = -1.0;
    //! used image size
    cv::Size _imgSize;

    //! calibration flags
    bool _fixAspectRatio        = true;
    bool _zeroTangentDistortion = true;
    bool _fixPrincipalPoint     = true;
    bool _zeroRadialDistortion  = true;

    //! if the calibration is scaled the used scale factor is stored here
    float _scaleFactor = 1.0f;
};

#endif //CAM_CALIBRATION_H
