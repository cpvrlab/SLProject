#ifndef CAMERA
#define CAMERA

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using namespace std;


class Camera
{
    public:
        Matrix4f extrinsic_matrix;
        Matrix3f intrinsic_matrix;
        Vector2f c;
        Vector2f K;
        Vector2f P;
        Vector3f pos;
        float fov;
        int width;
        int height;

        Camera();
        void look_at(Vector3f pos, Vector3f up, Vector3f point);
        void set_intrinsic(float fovy, int img_width, int img_height, float c0, float c1);
        void set_intrinsic(float fovy, int img_width, int img_height);
        void set_distortion(float k1, float k2, float p1, float p2);
};

#endif

