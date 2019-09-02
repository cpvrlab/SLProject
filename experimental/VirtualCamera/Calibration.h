#ifndef CALIBRATION
#define CALIBRATION

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using namespace std;

class Calibration
{
    public:

        static void init_matrix(Matrix3f &intrinsic_matrix, Vector2f &P, Vector2f &K, float fov, int width, int height);

        static float calibrate_opencv(Matrix3f &matrix, Vector2f &P, Vector2f &K, 
                                      int width, int height, 
                                      std::vector<std::vector<cv::Point2f>> &keypoints,
                                      std::vector<std::vector<cv::Point3f>> &worldpoints);

        static float calibrate_opencv(Matrix3f &intrinsic, 
                                      int width, int height, 
                                      std::vector<std::vector<cv::Point2f>> &keypoints,
                                      std::vector<std::vector<cv::Point3f>> &worldpoints);

        static float calibrate_opencv_fixed_center(Matrix3f &intrinsic, 
                                                   int width, int height, 
                                                   std::vector<std::vector<cv::Point2f>> &keypoints,
                                                   std::vector<std::vector<cv::Point3f>> &worldpoints);

        static float calibrate_brute_force(Matrix3f &matrix, Vector2f &P, Vector2f &K, 
                                           int width, int height, 
                                           std::vector<std::vector<cv::Point2f>> &keypoints,
                                           std::vector<std::vector<cv::Point3f>> &worldpoints);

        static float calibrate_unique_view(Matrix3f &matrix, Vector2f &P, Vector2f &K, 
                                           int width, int height, 
                                           std::vector<cv::Point2f> &keypoints,
                                           std::vector<cv::Point3f> &worldpoints);

        static float calibrate_ransac_unique_view(std::vector<Vector2f> &outliers, std::vector<Vector2f> &corrects,
                                                  Matrix3f &intrinsic, Vector2f &P, Vector2f &K, int width, int height, float &total_error, 
                                                  int nb_iter, int percent_correct, float threshold, int selection_percent, 
                                                  std::vector<cv::Point2f> keypoints, std::vector<cv::Point3f> worldpoints);

        static bool calibrate_ransac(Matrix3f& intrinsic, Vector2f &P, Vector2f &K, int width, int height, float& total_error, 
                                     int nb_iter, int percent_correct, float threshold, int nselect, 
                                     std::vector<std::vector<cv::Point2f>>& keypoints, 
                                     std::vector<std::vector<cv::Point3f>>& worldpoints);

    private:
        static float calibrate_unique_view(cv::Mat &matrix, cv::Mat &distortion, 
                                           cv::Size &size, 
                                           cv::Mat &rvec, cv::Mat &tvec, 
                                           std::vector<cv::Point2f> &keypoints,
                                           std::vector<cv::Point3f> &worldpoints);

        static float reprojectionRMS(cv::Mat intrinsic, cv::Mat distortion,
                                     std::vector<std::vector<cv::Point2f>>& vvP2D,
                                     std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                                     const std::vector<cv::Mat>& rvecs,
                                     const std::vector<cv::Mat>& tvecs);

        static float calibrate_brute_force(cv::Mat &intrinsic, 
                                           cv::Size &size, 
                                           std::vector<cv::Mat> &rvecs, 
                                           std::vector<cv::Mat> &tvecs,
                                           std::vector<std::vector<cv::Point2f>> &keypoints,
                                           std::vector<std::vector<cv::Point3f>> &worldpoints);

        static float calibrate_opencv_fixed_center(cv::Mat &matrix, 
                                                   cv::Size &size, 
                                                   std::vector<cv::Mat> &rvecs, 
                                                   std::vector<cv::Mat> &tvecs,
                                                   std::vector<std::vector<cv::Point2f>> &keypoints,
                                                   std::vector<std::vector<cv::Point3f>> &worldpoints);

        static void init_matrix(cv::Mat &mat, float fov, cv::Size &size);
};

#endif

