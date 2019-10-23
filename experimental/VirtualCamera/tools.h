#ifndef TOOLS
#define TOOLS
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "Camera.h"
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

std::vector<std::vector<bool>> gen_binary_vectors(std::vector<std::vector<cv::Point2f>> vvP2D);

std::vector<std::vector<cv::Point2f>> gen_selection_vectors(std::vector<std::vector<cv::Point2f>> vvP2D);

std::vector<std::vector<cv::Point3f>> gen_selection_vectors(std::vector<std::vector<cv::Point3f>> vvP3Dw);

void select_random(std::vector<bool>& selection, int n);
    
void select_random(std::vector<std::vector<bool>>& selections, int n);

void pick_selection(std::vector<cv::Point2f>& skp, 
                    std::vector<cv::Point3f>& swp, 
                    std::vector<cv::Point2f>& nskp, 
                    std::vector<cv::Point3f>& nswp, 
                    std::vector<bool>& selection, 
                    std::vector<cv::Point2f>& keypoints, 
                    std::vector<cv::Point3f>& worldpoints);

void pick_selection(std::vector<std::vector<cv::Point2f>>& skp, 
                    std::vector<std::vector<cv::Point3f>>& swp, 
                    std::vector<std::vector<cv::Point2f>>& nskp, 
                    std::vector<std::vector<cv::Point3f>>& nswp, 
                    std::vector<std::vector<bool>>& selections, 
                    std::vector<std::vector<cv::Point2f>>& keypoints, 
                    std::vector<std::vector<cv::Point3f>>& worldpoints);

Vector3f distord(Camera * c, Vector3f p);

Vector2f project_point(Camera * c, Vector3f p);

std::vector<Vector3f> generate_cloud(int nb_points, float w, float d, float h);

std::vector<Vector3f> get_visible_points(Camera * c, std::vector<Vector3f> cloud);

std::vector<Vector2f> project_no_distortion(Camera * c, std::vector<Vector3f> visible_points);

std::vector<Vector2f> noise(std::vector<Vector2f> projected, float d);

std::vector<Vector3f> noise(std::vector<Vector3f> points, float d);

int random_subset(std::vector<bool> &subset, int size, float percent);

void randomize_subset(std::vector<Vector2f> &projected, std::vector<bool> &subset, float d);

std::vector<int> find_neighbors(std::vector<Vector2f> &projected, Vector2f point, float r);

void make_false_association(std::vector<Vector2f> &projected, std::vector<Vector3f> &points3d, float probability, float radius);

std::vector<Vector3f> camera_transform(Camera * c, std::vector<Vector3f> cloud);

std::vector<Vector2f> project_and_distord(Camera * c, std::vector<Vector3f> visible_points);

std::vector<Vector2f> noise_projection(std::vector<Vector2f> projected, float d);

void save_to(std::string name, std::vector<Vector2f> data);

void save_to(std::string name, std::vector<Vector3f> data);

void generate_view(std::vector<Vector3f> &visible_points, std::vector<Vector2f> &projected, std::vector<Vector3f> &cloud, Camera &c);

void load_view(std::vector<Vector3f> &visible_points, std::vector<Vector2f> &projected, std::string worldpoints_path, std::string keypoints_path);

cv::Mat eigen2cv(Matrix3f m);

Matrix3f cv2eigen(cv::Mat m);

float get_fovy(Matrix3f m);
float get_fovy(cv::Mat m);


#endif

