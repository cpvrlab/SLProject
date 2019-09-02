#ifndef WAIAUTOCALIBRATION
#define WAIAUTOCALIBRATION
using namespace std;
#include <WAICalibration.h>
#include <WAISensorCamera.h>
#include <opencv2/core/core.hpp>

class AutoCalibration : public WAICalibration
{
    public:
    AutoCalibration(int width, int height);

    void  reset();
    bool  tryCalibrateRansac(std::vector<cv::Point2f> vP2D, std::vector<cv::Point3f> vP3Dw);
    bool  tryCalibrateBruteForce(std::vector<cv::Point2f> vP2D, std::vector<cv::Point3f> vP3Dw);
    void  setCameraParameters(float fx, float fy, float cx, float cy, float k1 = 0, float k2 = 0, float p1 = 0, float p2 = 0);
    float getError() { return _error; }

    private:
    void savePoints();

    bool calibrate_ransac(cv::Mat& intrinsic, cv::Mat& distortion, cv::Size& size, float& total_error, int nb_iter, int percent_correct, float threshold, int nselect, std::vector<cv::Point2f>& keypoints, std::vector<cv::Point3f>& worldpoints);

    bool calibrate_ransac(cv::Mat& intrinsic, cv::Mat& distortion, cv::Size& size, float& total_error, int nb_iter, int percent_correct, float threshold, int nselect, std::vector<std::vector<cv::Point2f>>& keypoints, std::vector<std::vector<cv::Point3f>>& worldpoints);

    void mean_position(cv::Point3f& mean, cv::Point3f& max, cv::Point3f& min, std::vector<cv::Point3f>& points3d);

    void select_random(std::vector<bool>& selection, int n);
    void select_random(std::vector<std::vector<bool>>& selections, int n);

    void pick_selection(std::vector<cv::Point2f>& skp, std::vector<cv::Point3f>& swp, std::vector<cv::Point2f>& nskp, std::vector<cv::Point3f>& nswp, std::vector<bool>& selection, std::vector<cv::Point2f>& keypoints, std::vector<cv::Point3f>& worldpoints);

    void pick_selection(std::vector<std::vector<cv::Point2f>>& skp,
                        std::vector<std::vector<cv::Point3f>>& swp,
                        std::vector<std::vector<cv::Point2f>>& nskp,
                        std::vector<std::vector<cv::Point3f>>& nswp,
                        std::vector<std::vector<bool>>&        selections,
                        std::vector<std::vector<cv::Point2f>>& keypoints,
                        std::vector<std::vector<cv::Point3f>>& worldpoints);

    float calibrate_opencv(cv::Mat& matrix, cv::Mat& distortion, cv::Size& size, cv::Mat& rvec, cv::Mat& tvec, std::vector<cv::Point2f>& keypoints, std::vector<cv::Point3f>& worldpoints);

    float calibrate_opencv(cv::Mat& matrix, cv::Mat& distortion, cv::Size& size, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs, std::vector<std::vector<cv::Point2f>>& keypoints, std::vector<std::vector<cv::Point3f>>& worldpoints);

    float calibrate_opencv_no_distortion(cv::Mat& matrix, cv::Size& size, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs, std::vector<std::vector<cv::Point2f>>& keypoints, std::vector<std::vector<cv::Point3f>>& worldpoints);

    float calibrate_opencv_no_distortion_fixed_center(cv::Mat& matrix, cv::Size& size,
                                                     std::vector<cv::Mat>& rvecs,
                                                     std::vector<cv::Mat>& tvecs,
                                                     std::vector<std::vector<cv::Point2f>>& keypoints,
                                                     std::vector<std::vector<cv::Point3f>>& worldpoints);

    float reprojectionRMS(cv::Mat intrinsic, cv::Mat distortion,
                          std::vector<cv::Point2f>& vP2D,
                          std::vector<cv::Point3f>& vP3Dw,
                          const cv::Mat& rvec,
                          const cv::Mat& tvec);

    float reprojectionRMS(cv::Mat intrinsic, cv::Mat distortion,
                          std::vector<std::vector<cv::Point2f>>& vvP2D,
                          std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                          const std::vector<cv::Mat>& rvecs,
                          const std::vector<cv::Mat>& tvecs);

    bool calibrateBruteForce(cv::Mat &intrinsic,
                             std::vector<std::vector<cv::Point2f>>& vvP2D,
                             std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                             std::vector<cv::Mat>& rvecs,
                             std::vector<cv::Mat>& tvecs,
                             float &error);

    cv::Mat                               _intrinsic;
    cv::Mat                               _distortion;
    std::vector<std::vector<cv::Point2f>> _vvP2D;
    std::vector<std::vector<cv::Point3f>> _vvP3Dw;
    float                                 _error;
};
#endif
