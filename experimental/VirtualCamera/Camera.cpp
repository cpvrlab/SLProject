#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include "Camera.h"

Camera::Camera()
{
}

void Camera::look_at(Vector3f pos, Vector3f up, Vector3f point)
{
    this->pos = pos;
    Vector3f dir = (point - pos).normalized();
    Vector3f right = dir.cross(up).normalized();
    up = right.cross(dir).normalized();

    extrinsic_matrix << right[0], right[1], right[2], -right.dot(pos),
                     up[0], up[1], up[2], -up.dot(pos),
                     -dir[0], -dir[1], -dir[2], dir.dot(pos),
                     0, 0, 0, 1;
}

void Camera::set_intrinsic(float fovy, int img_width, int img_height, float c0, float c1)
{
    fov = fovy;
    float fov_rad = fovy * M_PI / 180.0;
    c[0] = c0;
    c[1] = c1;
    float fy = c[1] / tanf(fov_rad * 0.5f);
    float fx = fy;
    width = img_width;
    height = img_height;

    intrinsic_matrix << fx, 0 , c[0],
    0,  fy, c[1],
    0,  0 , 1;
}

void Camera::set_intrinsic(float fovy, int img_width, int img_height)
{
    c[0] = img_width * 0.5f;
    c[1] = img_height * 0.5f;
    set_intrinsic(fovy, img_width, img_height, c[0], c[1]);
}

void Camera::set_distortion(float k1, float k2, float p1, float p2)
{
    K << k1, k2;
    P << p1, p2;
}

