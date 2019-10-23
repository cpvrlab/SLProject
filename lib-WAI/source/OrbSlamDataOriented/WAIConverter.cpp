#include "WAIConverter.h"

static g2o::SE3Quat convertCvMatToG2OSE3Quat(const cv::Mat& mat)
{
    Eigen::Matrix<r64, 3, 3> R;
    R << mat.at<r32>(0, 0), mat.at<r32>(0, 1), mat.at<r32>(0, 2),
      mat.at<r32>(1, 0), mat.at<r32>(1, 1), mat.at<r32>(1, 2),
      mat.at<r32>(2, 0), mat.at<r32>(2, 1), mat.at<r32>(2, 2);

    Eigen::Matrix<r64, 3, 1> t(mat.at<r32>(0, 3), mat.at<r32>(1, 3), mat.at<r32>(2, 3));

    g2o::SE3Quat result = g2o::SE3Quat(R, t);

    return result;
}

static cv::Mat convertG2OSE3QuatToCvMat(const g2o::SE3Quat se3)
{
    Eigen::Matrix<r64, 4, 4> mat = se3.to_homogeneous_matrix();

    cv::Mat result = cv::Mat(4, 4, CV_32F);
    for (i32 i = 0; i < 4; i++)
    {
        for (i32 j = 0; j < 4; j++)
        {
            result.at<r32>(i, j) = (r32)mat(i, j);
        }
    }

    return result;
}

static Eigen::Matrix<r64, 3, 1> convertCvMatToEigenVector3D(const cv::Mat& mat)
{
    Eigen::Matrix<r64, 3, 1> result;

    result << mat.at<r32>(0), mat.at<r32>(1), mat.at<r32>(2);

    return result;
}

static cv::Mat convertEigenVector3DToCvMat(const Eigen::Matrix<r64, 3, 1>& vec3d)
{
    cv::Mat result = cv::Mat(3, 1, CV_32F);

    for (int i = 0; i < 3; i++)
    {
        result.at<r32>(i) = (r32)vec3d(i);
    }

    return result;
}

static std::vector<cv::Mat> convertCvMatToDescriptorVector(const cv::Mat& descriptors)
{
    std::vector<cv::Mat> result;
    result.reserve(descriptors.rows);

    for (i32 j = 0; j < descriptors.rows; j++)
    {
        result.push_back(descriptors.row(j));
    }

    return result;
}