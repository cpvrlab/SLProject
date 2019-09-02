#ifndef VIEWSTORAGE
#define VIEWSTORAGE
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

class ViewStorage 
{
    public:
        
        std::vector<std::vector<cv::Point3f>> points3d;
        std::vector<std::vector<cv::Point2f>> points2d;

        void add_view(std::vector<Vector3f> position, std::vector<Vector2f> projected);
};

#endif

