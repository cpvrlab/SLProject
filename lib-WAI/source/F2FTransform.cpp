/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <F2FTransform.h>
#include <Random.h>

#include <OrbSlam/Optimizer.h>
#include <OrbSlam/ORBmatcher.h>
#include <Eigen/Geometry>
#include <thread>

void F2FTransform::opticalFlowMatch(const cv::Mat&             f1Gray,
                                    const cv::Mat&             f2Gray,
                                    std::vector<cv::KeyPoint>& kp1,
                                    std::vector<cv::Point2f>&  p1,
                                    std::vector<cv::Point2f>&  p2,
                                    std::vector<uchar>&        inliers,
                                    std::vector<float>&        err)
{
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);
    if (kp1.size() < 20)
        return;
    p1.clear();
    p2.clear();
    inliers.clear();
    err.clear();

    p1.reserve(kp1.size());
    p2.reserve(kp1.size());
    inliers.reserve(kp1.size());
    err.reserve(kp1.size());

    for (int i = 0; i < kp1.size(); i++)
        p1.push_back(kp1[i].pt);

    cv::Size winSize(11, 11);

    cv::calcOpticalFlowPyrLK(
      f1Gray,
      f2Gray,
      p1,                        // Previous and current keypoints coordinates.The latter will be
      p2,                        // expanded if more good coordinates are detected during OptFlow
      inliers,                   // Output vector for keypoint correspondences (1 = match found)
      err,                       // Error size for each flow
      winSize,                   // Search window for each pyramid level
      1,                         // Max levels of pyramid creation
      criteria,                  // Configuration from above
      0,                         // Additional flags
      0.001);                    // Minimal Eigen threshold
}

float F2FTransform::filterPoints(std::vector<cv::Point2f>& p1,
                                 std::vector<cv::Point2f>& p2,
                                 std::vector<cv::Point2f>& goodP1,
                                 std::vector<cv::Point2f>& goodP2,
                                 std::vector<uchar>&       inliers,
                                 std::vector<float>&       err)
{
    if (p1.size() == 0)
        return 0;
    goodP1.clear();
    goodP2.clear();
    goodP1.reserve(p1.size());
    goodP2.reserve(p1.size());
    float avgMotion = 0;

    for (int i = 0, j = 0; i < p1.size(); i++)
    {
        if (inliers[i] && err[i] < 5.0)
        {
            goodP1.push_back(p1[i]);
            goodP2.push_back(p2[i]);
            avgMotion += cv::norm(p1[i] - p2[i]);
            j++;
        }
    }
    return avgMotion / goodP1.size();
}

bool F2FTransform::estimateRot(const cv::Mat             K,
                               std::vector<cv::Point2f>& p1,
                               std::vector<cv::Point2f>& p2,
                               float&                    yaw,
                               float&                    pitch,
                               float&                    roll)
{
    if (p1.size() < 10)
        return false;

    cv::Mat H = estimateAffinePartial2D(p1, p2);
    float zrot = atan2(H.at<double>(1, 0), H.at<double>(0, 0));
    float dx = 0;//H.at<double>(0, 2);
    float dy = 0;//H.at<double>(1, 2);
    //Compute dx dy (estimageAffinePartial doesn't give right result when rotating on z axis)
    for (int i = 0; i < p1.size(); i++)
    {
        dx += p2[i].x - p1[i].x;
        dy += p2[i].y - p1[i].y;
    }
    dx /= (float)p1.size();
    dy /= (float)p1.size();

    Eigen::Vector3f v1(0, 0, 1.0);
    Eigen::Vector3f vx(dx, 0, K.at<double>(0, 0));
    Eigen::Vector3f vy(0, dy, K.at<double>(0, 0));
    vx.normalize();
    vy.normalize();

    float xrot = -acos(v1.dot(vx));
    float yrot = -acos(v1.dot(vy));

    roll = -zrot;

    if (vx.dot(Eigen::Vector3f::UnitX()) > 0)
        pitch = xrot;
    else
        pitch = -xrot;

    if (vy.dot(Eigen::Vector3f::UnitY()) > 0)
        yaw = yrot;
    else
        yaw = -yrot;


    return true;
}

void F2FTransform::eulerToMat(float yaw, float pitch, float roll, cv::Mat& Rx, cv::Mat& Ry, cv::Mat& Rz)
{
    Eigen::Matrix3f mx, my, mz;
    mx = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitX());
    my = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
    mz = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitZ());

    cv::Mat eRx = eigen2cv(mx);
    cv::Mat eRy = eigen2cv(my);
    cv::Mat eRz = eigen2cv(mz);

    Rx = cv::Mat::eye(4, 4, CV_32F);
    eRx.copyTo(Rx.rowRange(0, 3).colRange(0, 3));

    Ry = cv::Mat::eye(4, 4, CV_32F);
    eRy.copyTo(Ry.rowRange(0, 3).colRange(0, 3));

    Rz = cv::Mat::eye(4, 4, CV_32F);
    eRz.copyTo(Rz.rowRange(0, 3).colRange(0, 3));
}

cv::Mat F2FTransform::eigen2cv(Eigen::Matrix3f m)
{
    cv::Mat r;
    r = (cv::Mat_<float>(3, 3) << m(0), m(3), m(6), m(1), m(4), m(7), m(2), m(5), m(8));
    return r;
}
