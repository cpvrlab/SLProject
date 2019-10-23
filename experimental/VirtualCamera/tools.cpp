#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include "Camera.h"
#include "tools.h"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using namespace std;

std::vector<std::vector<bool>> gen_binary_vectors(std::vector<std::vector<cv::Point2f>> vvP2D)
{
    std::vector<std::vector<bool>> selections;
    selections.reserve(vvP2D.size());

    for (int i = 0; i < vvP2D.size(); i++)
    {
        std::vector<bool> binary_vector(vvP2D[i].size(), 0);
        selections.push_back(binary_vector);
    }
    return selections;
}

std::vector<std::vector<cv::Point2f>> gen_selection_vectors(std::vector<std::vector<cv::Point2f>> vvP2D)
{
    std::vector<std::vector<cv::Point2f>> selections;
    selections.reserve(vvP2D.size());

    for (int i = 0; i < vvP2D.size(); i++)
    {
        std::vector<cv::Point2f> v(vvP2D[i].size());
        selections.push_back(v);
    }
    return selections;
}

std::vector<std::vector<cv::Point3f>> gen_selection_vectors(std::vector<std::vector<cv::Point3f>> vvP3Dw)
{
    std::vector<std::vector<cv::Point3f>> selections;
    selections.reserve(vvP3Dw.size());

    for (int i = 0; i < vvP3Dw.size(); i++)
    {
        std::vector<cv::Point3f> v(vvP3Dw[i].size());
        selections.push_back(v);
    }
    return selections;
}

void select_random(std::vector<bool>& selection, int n)
{
    if (selection.size() <= n)
    {
        for (int i = 0; i < selection.size(); i++)
        {
            selection[i] = true;
        }
        return;
    }

    for (int i = 0; i < n; i++)
    {
        int idx = rand() % selection.size();
        if (selection[idx])
        {
            i--;
            continue;
        }
        selection[idx] = true;
    }
}

void select_random(std::vector<std::vector<bool>> &selections, int n)
{
    for (std::vector<bool> &selection : selections)
    {
        if (selection.size() <= n)
        {
            for (int i = 0; i < selection.size(); i++)
            {
                selection[i] = true;
            }
            return;
        }

        for (int i = 0; i < n; i++)
        {
            int idx = rand() % selection.size();
            if (selection[idx])
            {
                i--;
                continue;
            }
            selection[idx] = true;
        }
    }
}


void pick_selection(std::vector<cv::Point2f>& skp, 
                    std::vector<cv::Point3f>& swp, 
                    std::vector<cv::Point2f>& nskp, 
                    std::vector<cv::Point3f>& nswp, 
                    std::vector<bool>& selection, 
                    std::vector<cv::Point2f>& keypoints, 
                    std::vector<cv::Point3f>& worldpoints)
{
    for (int i = 0; i < selection.size(); i++)
    {
        if (selection[i])
        {
            skp.push_back(keypoints[i]);
            swp.push_back(worldpoints[i]);
            selection[i] = 0;
        }
        else
        {
            nskp.push_back(keypoints[i]);
            nswp.push_back(worldpoints[i]);
        }
    }
}


void pick_selection(std::vector<std::vector<cv::Point2f>>& skp, 
                    std::vector<std::vector<cv::Point3f>>& swp, 
                    std::vector<std::vector<cv::Point2f>>& nskp, 
                    std::vector<std::vector<cv::Point3f>>& nswp, 
                    std::vector<std::vector<bool>>& selections, 
                    std::vector<std::vector<cv::Point2f>>& keypoints, 
                    std::vector<std::vector<cv::Point3f>>& worldpoints)
{
    for (int i = 0; i < selections.size(); i++)
    {
        for (int j = 0; j < selections[i].size(); j++)
        {
            if (selections[i][j])
            {
                skp[i].push_back(keypoints[i][j]);
                swp[i].push_back(worldpoints[i][j]);
                selections[i][j] = 0;
            }
            else
            {
                nskp[i].push_back(keypoints[i][j]);
                nswp[i].push_back(worldpoints[i][j]);
            }
        }
    }
}

Vector3f distord(Camera * c, Vector3f p)
{
    Vector3f d;
    float r2 = p[0] * p[0] + p[1] * p[1];
    float r4 = r2 * r2;
    d = p * (1 + c->K[0] * r2 + c->K[1] * r4);
    d[0] += 2 * c->P[0] * p[0]*p[1] + c->P[1] * (r2 + 2 * p[0]*p[0]);
    d[1] += c->P[0] * (r2 + 2 * p[1]*p[1]) + 2 * c->P[1] * p[0]*p[1];
    d[2] = 1.0;
    return d;
}

Vector2f project_point(Camera * c, Vector3f p)
{
    Vector3f v = Vector3f (p[0] / p[2], p[1] / p[2], 1.0);
    Vector3f d = distord(c, v);
    Vector3f w = c->intrinsic_matrix * d;

    return Vector2f(w[0], w[1]);
}

std::vector<Vector3f> generate_cloud(int nb_points, float w, float d, float h)
{
    std::vector<Vector3f> cloud;

    for (int i = 0; i < nb_points; i++)
    {
        Vector3f p;
        p[0] = w * ((float)rand() / (float)RAND_MAX - 0.5);
        p[1] = h * ((float)rand() / (float)RAND_MAX - 0.5);
        p[2] = d * ((float)rand() / (float)RAND_MAX - 0.5);
        cloud.push_back(p);
    }
    return cloud;
}

std::vector<Vector3f> get_visible_points(Camera * c, std::vector<Vector3f> cloud)
{
    std::vector<Vector3f> visible_points;

    for (Vector3f p : cloud)
    {
        Vector4f v = (c->extrinsic_matrix * Vector4f(p[0], p[1], p[2], 1.0));
        Vector3f w = Vector3f(v[0], v[1], v[2]);

        if (v[2] <= 0.0000001)
        {
            Vector2f u = project_point(c, w);
            if (u[0] > 0 && u[1] > 0 && u[0] < c->width && u[1] < c->height)
                visible_points.push_back(p);
        }
    }
    return visible_points;
}

std::vector<Vector2f> project_no_distortion(Camera * c, std::vector<Vector3f> visible_points)
{
    std::vector<Vector2f> projected;

    for (Vector3f p : visible_points)
    {
        Vector3f v = Vector3f (p[0] / p[2], p[1] / p[2], 1.0);
        Vector3f w = c->intrinsic_matrix * v;
        projected.push_back(Vector2f(w[0], w[1]));
    }
    return projected;
}

std::vector<Vector2f> noise(std::vector<Vector2f> projected, float d)
{
    std::vector<Vector2f> noised;
    for (Vector2f p : projected)
    {
        float radius = 2. * d * ((float)rand() / (float)RAND_MAX - 0.5);
        float angle = 2. * M_PI * ((float)rand() / (float)RAND_MAX - 0.5); 
        Vector2f n = p + radius * Vector2f(cos(angle), sin(angle));
        noised.push_back(n);
    }
    return noised;
}

std::vector<Vector3f> noise(std::vector<Vector3f> points, float d)
{
    std::vector<Vector3f> noised;
    for (Vector3f p : points)
    {
        float radius = 2. * d * ((float)rand() / (float)RAND_MAX - 0.5);
        float sigma = 2. * M_PI * ((float)rand() / (float)RAND_MAX - 0.5); 
        float theta = 2. * M_PI * ((float)rand() / (float)RAND_MAX - 0.5); 
        float stheta = sin(theta);
        float ctheta = cos(theta);
        float ssigma = sin(sigma);
        float csigma = cos(sigma);
        Vector3f n = p + radius * Vector3f(radius * stheta * csigma, radius * stheta * ssigma, radius * ctheta);
        noised.push_back(n);
    }
    return noised;
}

void randomize_subset(std::vector<Vector2f> &projected, std::vector<bool> &subset, float d)
{
    for (int i = 0; i < subset.size(); i++)
    {
        bool b = subset[i];
        if (b == true)
        {
            float radius = 2. * d * ((float)rand() / (float)RAND_MAX - 0.5);
            float angle = 2. * M_PI * ((float)rand() / (float)RAND_MAX - 0.5); 

            Vector2f p = projected[i];
            projected[i] = p + radius * Vector2f(cos(angle), sin(angle));
        }
    }
}

std::vector<int> find_neighbors(std::vector<Vector2f> &projected, Vector2f point, float r)
{
    std::vector<int> neighbors;

    for (int i = 0; i < projected.size(); i++)
    {
        Vector2f n = projected[i];
        if ((n - point).norm() <= r)
            neighbors.push_back(i);
    }
    return neighbors;
}

void make_false_association(std::vector<Vector2f> &projected, std::vector<Vector3f> &points3d, float probability, float radius)
{
    for (int i = 0; i < projected.size(); i++)
    {
        if (((float)rand() / (float)RAND_MAX) <= probability)
        {
            Vector2f p = projected.at(i);
            std::vector<int> neighbors = find_neighbors(projected, p, radius);
            if (neighbors.size() == 0)
                continue;

            // Get a random idx from the neighbors
            int idx = (rand() * neighbors.size()) / RAND_MAX; 

            //Swap point at i and idx
            Vector2f v2 = projected[i];
            projected[i] = projected[idx];
            projected[idx] = v2;

            Vector3f v3 = points3d[i];
            points3d[i] = points3d[idx];
            points3d[idx] = v3;
        }
    }
}

std::vector<Vector3f> camera_transform(Camera * c, std::vector<Vector3f> cloud)
{
    std::vector<Vector3f> transformed_points;

    for (Vector3f p : cloud)
    {
        Vector4f v = (c->extrinsic_matrix * Vector4f(p[0], p[1], p[2], 1.0));
        Vector3f w = Vector3f(v[0], v[1], v[2]);
        transformed_points.push_back(w);
    }
    return transformed_points;
}

std::vector<Vector2f> project_and_distord(Camera * c, std::vector<Vector3f> visible_points)
{
    std::vector<Vector2f> projected;

    for (Vector3f p : visible_points)
    {
        Vector3f v = Vector3f (p[0] / p[2], p[1] / p[2], 1.0);
        Vector3f d = distord(c, v);
        Vector3f w = c->intrinsic_matrix * d;

        projected.push_back(Vector2f(w[0], w[1]));
    }
    return projected;
}

std::vector<Vector2f> noise_projection(std::vector<Vector2f> projected, float d)
{
    std::vector<Vector2f> noised;
    for (Vector2f p : projected)
    {
        float radius = 2. * d * ((float)rand() / (float)RAND_MAX - 0.5);
        float angle = 2. * M_PI * ((float)rand() / (float)RAND_MAX - 0.5); 
        Vector2f n = p + radius * Vector2f(cos(angle), sin(angle));
        noised.push_back(n);
    }
    return noised;
}

void save_to(std::string name, std::vector<Vector2f> data)
{
    ofstream file;
    file.open(name);

    for (Vector2f p : data)
        file << p[0] << " " << p[1] << endl;
    file.close();
}

void save_to(std::string name, std::vector<Vector3f> data)
{
    ofstream file;
    file.open(name);

    for (Vector3f p : data)
        file << p[0] << " " << p[1] << " " << p[2] << endl;
    file.close();
}

void generate_view(std::vector<Vector3f> &visible_points, std::vector<Vector2f> &projected, std::vector<Vector3f> &cloud, Camera &c)
{
    float noise_radius = 7; //in px
    float false_association_radius = 5;
    float false_association_probability = 0.5;
    std::vector<Vector3f> transformed_points;
    std::vector<Vector2f> noised;

    visible_points = get_visible_points(&c, cloud);

    transformed_points = camera_transform(&c, visible_points);
    projected = project_and_distord(&c, transformed_points);

    //40% of points will be really wrong
    std::vector<bool> subst;
    select_random(subst, (projected.size() * 40) / 100);
    randomize_subset(projected, subst, 20);
}

static void trim(std::string &str)
{
    str.erase(0, str.find_first_not_of("[\n\r\t, "));
    str.erase(str.find_last_not_of("]\n\r\t, ")+1);
}

void load_view(std::vector<Vector3f> &visible_points, std::vector<Vector2f> &projected, std::string worldpoints_path, std::string keypoints_path)
{
    ifstream myfile;
    myfile.open(worldpoints_path);
    for(std::string line; getline(myfile, line);)
    {
        float x, y, z;
        size_t idx;
        trim(line);

        x = stof(line, &idx);
        line = line.substr(idx);
        trim(line);

        y = stof(line, &idx);
        line = line.substr(idx);
        trim(line);

        z = stof(line, &idx);

        visible_points.push_back(Vector3f(x, y, z));

    }
    myfile.close();

    myfile.open(keypoints_path);
    for(std::string line; getline(myfile, line);)
    { 
        float x, y;
        size_t idx;
        trim(line);

        x = stof(line, &idx);
        line = line.substr(idx);
        trim(line);

        y = stof(line, &idx);
        line = line.substr(idx);
        trim(line);

        projected.push_back(Vector2f(x, y));

    }
    myfile.close();
}

cv::Mat eigen2cv(Matrix3f m)
{
    cv::Mat r;
    r = (cv::Mat_<double>(3, 3) << m(0), m(3), m(6), m(1), m(4), m(7), m(2), m(5), m(8));
    return r;
}

Matrix3f cv2eigen(cv::Mat m)
{
    Matrix3f r;

    r << m.at<double>(0, 0), m.at<double>(0,1), m.at<double>(0,2), m.at<double>(1,0), m.at<double>(1,1), m.at<double>(1,2), m.at<double>(2,0), m.at<double>(2,1), m.at<double>(2,2);
    return r;
}

float get_fovy(Matrix3f m)
{
    float fy     = m(4);
    float cy     = m(7);
    float fov    = 2.0 * atan2(cy, fy);
    return fov * 180.0 / M_PI;
}

float get_fovy(cv::Mat m)
{
    float fy  = m.at<double>(1, 1);
    float cy  = m.at<double>(1, 2);
    float fov = 2.0 * atan2(cy, fy);
    return fov * 180.0 / M_PI;
}

