#ifndef WAIAUTOCALIBRATION
#define WAIAUTOCALIBRATION

#include <WAICalibration.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <thread>
#include <mutex>
#include <sens/SENSCalibration.h>

using namespace std;

class AutoCalibration
{
public:
    AutoCalibration(cv::Size frameSize, float mapDimension);
    ~AutoCalibration();

    static void calibrateFrames(AutoCalibration* ac);

    bool fillFrame(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>& matching,
                   cv::Mat                                                        tcw);

    bool hasSucceed();
    bool hasCalibration();

    const SENSCalibration& consumeCalibration();
    void                   reset();

    static bool calibrateBruteForce(cv::Mat&                               intrinsic,
                                    std::vector<std::vector<cv::Point2f>>& vvP2D,
                                    std::vector<std::vector<cv::Point3f>>& vvP3Dw,
                                    std::vector<cv::Mat>&                  rvecs,
                                    std::vector<cv::Mat>&                  tvecs,
                                    cv::Size                               size,
                                    float&                                 error);

    static bool calibrateBruteForce(cv::Mat&                  intrinsic,
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
                                         int                                    nb_iter,
                                         float                                  threshold,
                                         int                                    iniModelSize,
                                         std::vector<std::vector<cv::Point2f>>& keypoints,
                                         std::vector<std::vector<cv::Point3f>>& worldpoints,
                                         int                                    nbVectors);

    static void computeMatrix(cv::Size size, cv::Mat& mat, cv::Mat& distortion, float fov);
    static void select_random(std::vector<bool>& selection, int n);
    static void select_random(std::vector<std::vector<bool>>& selections, int n);

    static void pick_points(std::vector<bool>&        selections,
                            std::vector<cv::Point2f>& skp,
                            std::vector<cv::Point3f>& swp,
                            std::vector<cv::Point2f>& nskp,
                            std::vector<cv::Point3f>& nswp,
                            std::vector<cv::Point2f>& keypoints,
                            std::vector<cv::Point3f>& worldpoints);

    static void pick_frames(std::vector<bool>&                     selections,
                            std::vector<std::vector<cv::Point2f>>& skp,
                            std::vector<std::vector<cv::Point3f>>& swp,
                            std::vector<std::vector<cv::Point2f>>& nskp,
                            std::vector<std::vector<cv::Point3f>>& nswp,
                            std::vector<std::vector<cv::Point2f>>& keypoints,
                            std::vector<std::vector<cv::Point3f>>& worldpoints);

    static float calibrate_opencv(cv::Mat&                               matrix,
                                  cv::Mat&                               distortion,
                                  cv::Size&                              size,
                                  std::vector<cv::Mat>&                  rvecs,
                                  std::vector<cv::Mat>&                  tvecs,
                                  std::vector<std::vector<cv::Point2f>>& keypoints,
                                  std::vector<std::vector<cv::Point3f>>& worldpoints);

    static bool calibrate(SENSCalibration*&                                                           calibration,
                          cv::Size                                                                    size,
                          std::vector<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>>& matchings);

private:
    static float calcCameraVerticalFOV(cv::Mat& cameraMat);
    static float calcCameraHorizontalFOV(cv::Mat& cameraMat);
    static void  genIntrinsicMatrix(int width, int height, cv::Mat& mat, float fov);

    std::vector<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>> _calibrationMatchings;

    std::vector<cv::Mat> _framesDir;
    std::vector<cv::Mat> _framesPos;
    float                _mapDimension;
    std::thread          _calibrationThread;
    cv::Size             _frameSize;
    SENSCalibration*     _calibration = nullptr;
    bool                 _hasCalibration;
    bool                 _isRunning;
    bool                 _isFinished;
    std::mutex           _calibrationMutex;
};
#endif
