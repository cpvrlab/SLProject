
#ifndef F2FTRANSFORM
#define F2FTRANSFORM

#include <opencv2/opencv.hpp>

#include <WAIHelper.h>
#include <WAIFrame.h>
#include <Eigen/Core>

class WAI_API F2FTransform
{
public:
    static void opticalFlowMatch(const cv::Mat&            f1Gray,
                                 const cv::Mat&            f2Gray,
                                 std::vector<cv::Point2f>& p1,
                                 std::vector<cv::Point2f>& p2,
                                 std::vector<uchar>&       inliers,
                                 std::vector<float>&       err);

    static float filterPoints(const std::vector<cv::Point2f>& p1,
                              const std::vector<cv::Point2f>& p2,
                              std::vector<cv::Point2f>&       goodP1,
                              std::vector<cv::Point2f>&       goodP2,
                              std::vector<uchar>&             inliers,
                              std::vector<float>&             err);

    static bool estimateRot(const cv::Mat             K,
                            std::vector<cv::Point2f>& p1,
                            std::vector<cv::Point2f>& p2,
                            float&                    yaw,
                            float&                    pitch,
                            float&                    roll);

    static bool estimateRotXYZ(const cv::Mat&                  K,
                               const std::vector<cv::Point2f>& p1,
                               const std::vector<cv::Point2f>& p2,
                               float&                          xAngRAD,
                               float&                          yAngRAD,
                               float&                          zAngRAD,
                               std::vector<uchar>&             inliers);

    static bool estimateRotXY(const cv::Mat&                  K,
                              const std::vector<cv::Point2f>& p1,
                              const std::vector<cv::Point2f>& p2,
                              float&                          xAngRAD,
                              float&                          yAngRAD,
                              const float                     zAngRAD,
                              std::vector<uchar>&             inliers);

    static void eulerToMat(float xAngRAD, float yAngRAD, float zAngRAD, cv::Mat& Rx, cv::Mat& Ry, cv::Mat& Rz);

private:
    static cv::Mat eigen2cv(Eigen::Matrix3f m);
};

#endif
