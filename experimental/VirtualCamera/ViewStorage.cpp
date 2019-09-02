#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ViewStorage.h"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using namespace std;

void ViewStorage::add_view(std::vector<Vector3f> position, std::vector<Vector2f> projected)
{
    std::vector<cv::Point3f> cv_position;
    std::vector<cv::Point2f> cv_projected;

    for (Vector3f v : position)
        cv_position.push_back(cv::Point3f(v[0], v[1], v[2]));

    for (Vector2f v : projected)
        cv_projected.push_back(cv::Point2f(v[0], v[1]));

    points3d.push_back(cv_position);
    points2d.push_back(cv_projected);
}

