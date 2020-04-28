#ifndef WAIAUTOCALIBRATION
#define WAIAUTOCALIBRATION

using namespace std;

#include <WAICalibration.h>
#include <vector>
#include <deque>

#include <opencv2/core/core.hpp>

class AutoCalibration
{
    public:
        static void calibrateBruteForce(cv::Mat&                                intrinsic,
                                        cv::Mat&                                distortion,
                                        std::vector<std::vector<cv::Point2f>>& vvP2D,
                                        std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                                        std::vector<cv::Mat>&                   rvecs,
                                        std::vector<cv::Mat>&                   tvecs,
                                        cv::Size                                size,
                                        float&                                  error);

        static void calibrateBruteForce(cv::Mat&                  intrinsic,
                                        cv::Mat&                  distortion,
                                        std::vector<cv::Point2f>& vP2D,
                                        std::vector<cv::Point3f>& vP3Dw,
                                        cv::Mat&                  rvec,
                                        cv::Mat&                  tvec,
                                        cv::Size                  size,
                                        float&                    error);

        static float ransac_frame_points(cv::Size&                 size,
                                         int                       nbIter,
                                         float                     threshold,
                                         int                       iniModelSize,
                                         std::vector<cv::Point2f>& keypoints,
                                         std::vector<cv::Point3f>& worldpoints,
                                         std::vector<cv::Point2f>& outKeypoints,
                                         std::vector<cv::Point3f>& outWorldpoints);

        static float calibrate_frames_ransac(cv::Size&                              size,
                                             cv::Mat&                               intrinsic,
                                             cv::Mat&                               distortion,
                                             int                                    nb_iter,
                                             float                                  threshold,
                                             int                                    iniModelSize,
                                             std::vector<std::vector<cv::Point2f>>& keypoints,
                                             std::vector<std::vector<cv::Point3f>>& worldpoints);

        static void computeMatrix(cv::Size size, cv::Mat& mat, float fov);
        static void select_random(std::vector<bool>& selection, int n);
        static void select_random(std::vector<std::vector<bool>>& selections, int n);

        static void pick_points(std::vector<bool>&        selections,
                                std::vector<cv::Point2f>& skp,
                                std::vector<cv::Point3f>& swp,
                                std::vector<cv::Point2f>& nskp,
                                std::vector<cv::Point3f>& nswp,
                                std::vector<cv::Point2f>& keypoints,
                                std::vector<cv::Point3f>& worldpoints);

        static void pick_frames(std::vector<bool>&                      selections,
                                std::vector<std::vector<cv::Point2f>>& skp,
                                std::vector<std::vector<cv::Point3f>>& swp,
                                std::vector<std::vector<cv::Point2f>>& nskp,
                                std::vector<std::vector<cv::Point3f>>& nswp,
                                std::vector<std::vector<cv::Point2f>>&  keypoints,
                                std::vector<std::vector<cv::Point3f>>&  worldpoints);

        static float calibrate_opencv(cv::Mat&                                matrix,
                                      cv::Mat&                                distortion,
                                      cv::Size&                               size,
                                      std::vector<cv::Mat>&                   rvecs,
                                      std::vector<cv::Mat>&                   tvecs,
                                      std::vector<std::vector<cv::Point2f>>& keypoints,
                                      std::vector<std::vector<cv::Point3f>>& worldpoints);

        static void calibrate(cv::Size&                                                                  size,
                              std::deque<std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>>> matchings);

    private:

        static float calcCameraVerticalFOV(cv::Mat& cameraMat);
        static float calcCameraHorizontalFOV(cv::Mat& cameraMat);
        static void genIntrinsicMatrix(int width, int height, cv::Mat& mat, float fov);
        static void computeMatrix(cv::Mat& mat, float fov);
};
#endif
