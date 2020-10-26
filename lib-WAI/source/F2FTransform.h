
#ifndef F2FTRANSFORM 
#define F2FTRANSFORM

#include <opencv2/opencv.hpp>

#include <WAIHelper.h>
#include <WAIFrame.h>
#include <Eigen/Core>

class WAI_API F2FTransform
{
public:
    static void opticalFlowMatch(const cv::Mat&             f1Gray,
                                 const cv::Mat&             f2Gray,
                                 std::vector<cv::KeyPoint>& kp1,
                                 std::vector<cv::Point2f>&  p1,
                                 std::vector<cv::Point2f>&  p2,
                                 std::vector<uchar>&        inliers,
                                 std::vector<float>&        err);

    static float filterPoints(std::vector<cv::Point2f>& p1,
                              std::vector<cv::Point2f>& p2,
                              std::vector<cv::Point2f>& goodP1,
                              std::vector<cv::Point2f>& goodP2,
                              std::vector<uchar>&       inliers,
                              std::vector<float>&       err);

    static bool estimateRot(const cv::Mat             K,
                            std::vector<cv::Point2f>& p1,
                            std::vector<cv::Point2f>& p2,
                            float&                    yaw,
                            float&                    pitch,
                            float&                    roll);

    static void eulerToMat(float yaw, float pitch, float roll, cv::Mat& Rx, cv::Mat& Ry, cv::Mat& Rz);

private:
    static cv::Mat eigen2cv(Eigen::Matrix3f m);
};

#endif
