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

#include <orb_slam/Optimizer.h>
#include <orb_slam/ORBmatcher.h>
#include <Eigen/Geometry>
#include <thread>
#include <Utils.h>

void F2FTransform::opticalFlowMatch(const cv::Mat&            f1Gray,
                                    const cv::Mat&            f2Gray,
                                    std::vector<cv::Point2f>& p1, //last
                                    std::vector<cv::Point2f>& p2, //curr
                                    std::vector<uchar>&       inliers,
                                    std::vector<float>&       err)
{
    if (p1.size() < 10)
        return;

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

    p2.clear();
    inliers.clear();
    err.clear();

    p2.reserve(p1.size());
    inliers.reserve(p1.size());
    err.reserve(p1.size());

    cv::Size winSize(11, 11);

    cv::calcOpticalFlowPyrLK(
      f1Gray,
      f2Gray,
      p1,       // Previous and current keypoints coordinates.The latter will be
      p2,       // expanded if more good coordinates are detected during OptFlow
      inliers,  // Output vector for keypoint correspondences (1 = match found)
      err,      // Error size for each flow
      winSize,  // Search window for each pyramid level
      1,        // Max levels of pyramid creation
      criteria, // Configuration from above
      0,        // Additional flags
      0.001);   // Minimal Eigen threshold
}

float F2FTransform::filterPoints(const std::vector<cv::Point2f>& p1,
                                 const std::vector<cv::Point2f>& p2,
                                 std::vector<cv::Point2f>&       goodP1,
                                 std::vector<cv::Point2f>&       goodP2,
                                 std::vector<uchar>&             inliers,
                                 std::vector<float>&             err)
{
    if (p1.size() == 0)
        return 0;
    goodP1.clear();
    goodP2.clear();
    goodP1.reserve(p1.size());
    goodP2.reserve(p1.size());
    float avgMotion = 0;

    for (int i = 0; i < p1.size(); i++)
    {
        if (inliers[i] && err[i] < 10.0)
        {
            goodP1.push_back(p1[i]);
            goodP2.push_back(p2[i]);
            avgMotion += (float)cv::norm(p1[i] - p2[i]);
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

    cv::Mat H    = estimateAffinePartial2D(p1, p2);
    float   zrot = (float)atan2(H.at<double>(1, 0), H.at<double>(0, 0));
    float   dx   = 0; //H.at<double>(0, 2);
    float   dy   = 0; //H.at<double>(1, 2);
    //Compute dx dy (estimageAffinePartial doesn't give right result when rotating on z axis)
    for (int i = 0; i < p1.size(); i++)
    {
        dx += p2[i].x - p1[i].x;
        dy += p2[i].y - p1[i].y;
    }
    dx /= (float)p1.size();
    dy /= (float)p1.size();

    Eigen::Vector3f v1(0, 0, 1.0);
    Eigen::Vector3f vx(dx, 0, (float)K.at<double>(0, 0));
    Eigen::Vector3f vy(0, dy, (float)K.at<double>(0, 0));
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

bool F2FTransform::estimateRotXYZ(const cv::Mat&                  K,
                                  const std::vector<cv::Point2f>& p1,
                                  const std::vector<cv::Point2f>& p2,
                                  float&                          xAngRAD,
                                  float&                          yAngRAD,
                                  float&                          zAngRAD,
                                  std::vector<uchar>&             inliers)
{
    if (p1.size() < 10)
        return false;

    //relate points to optical center
    //HINT: void at(int row, int column)
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    cv::Point2f              c((float)cx, (float)cy); //optical center
    std::vector<cv::Point2f> p1C = p1;
    std::vector<cv::Point2f> p2C = p2;
    for (int i = 0; i < p1.size(); i++)
    {
        p1C[i] -= c;
        p2C[i] -= c;
    }

    inliers.clear();
    //estimate homography (gives us frame b w.r.t frame a)
    cv::Mat aHb = cv::estimateAffinePartial2D(p1C, p2C, inliers);
    if (aHb.empty())
        return false;
    //std::cout << "aHb: " << aHb << std::endl;

    //express translational part in aHb relative to a coordinate frame b' that was rotated with rotational part
    cv::Mat bRa = aHb.rowRange(0, 2).colRange(0, 2).t(); //extract and express rotation from rotated frame
    cv::Mat aTR = aHb.col(2);
    cv::Mat bTR = bRa * aTR;

    //rotation about z-axis: It points into the image plane, so we have to invert the sign
    zAngRAD = (float)atan2(aHb.at<double>(1, 0), aHb.at<double>(0, 0));
    //rotation around y-axis about x-offset: it points down in cv image, so we have to invert the sign
    yAngRAD = (float)atan(bTR.at<double>(0) / K.at<double>(0, 0));
    //rotation around x-axis about y-offset: it points down in cv image, so we have to invert the sign
    xAngRAD = (float)atan(bTR.at<double>(1) / K.at<double>(1, 1));

    return true;
}

bool F2FTransform::estimateRotXY(const cv::Mat&                  K,
                                 const std::vector<cv::Point2f>& p1,
                                 const std::vector<cv::Point2f>& p2,
                                 float&                          xAngRAD,
                                 float&                          yAngRAD,
                                 const float                     zAngRAD,
                                 std::vector<uchar>&             inliers)
{
    if (p1.size() < 10)
        return false;

    //relate points to optival center
    //HINT: void at(int row, int column)
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    //rotation matrix

    cv::Point2f              c((float)cx, (float)cy); //optical center
    std::vector<cv::Point2f> p1C = p1;
    std::vector<cv::Point2f> p2C = p2;
    for (int i = 0; i < p1.size(); i++)
    {
        p1C[i] -= c;
        p2C[i] -= c;
        //rotate points about center
    }

    //estimate median shift

    inliers.clear();
    //estimate homography (gives us frame b w.r.t frame a)
    cv::Mat aHb = cv::estimateAffinePartial2D(p1C, p2C, inliers);
    if (aHb.empty())
        return false;
    //std::cout << "aHb: " << aHb << std::endl;

    //express translational part in aHb relative to a coordinate frame b' that was rotated with rotational part
    //cv::Mat bRa = aHb.rowRange(0, 2).colRange(0, 2).t(); //extract and express rotation from rotated frame
    //cv::Mat aTR = aHb.col(2);
    //cv::Mat bTR = bRa * aTR;

    //rotation about z-axis: It points into the image plane, so we have to invert the sign
    //zAngRAD = atan2(aHb.at<double>(1, 0), aHb.at<double>(0, 0));
    //rotation around y-axis about x-offset: it points down in cv image, so we have to invert the sign
    yAngRAD = (float)atan(aHb.at<double>(0, 2) / K.at<double>(0, 0));
    //rotation around x-axis about y-offset: it points down in cv image, so we have to invert the sign
    xAngRAD = (float)atan(aHb.at<double>(1, 2) / K.at<double>(1, 1));

    //std::cout << "xoff: "<< bTR.at<double>(0) << std::endl;
    //std::cout << "yoff: "<< bTR.at<double>(1) << std::endl;

    //std::cout << "zAngDEG: "<< zAngDEG << std::endl;
    //std::cout << "yAngDEG: "<< yAngDEG << std::endl;
    //std::cout << "xAngDEG: "<< xAngDEG << std::endl;

    return true;
}

void F2FTransform::eulerToMat(float xAngRAD, float yAngRAD, float zAngRAD, cv::Mat& Rx, cv::Mat& Ry, cv::Mat& Rz)
{
    Eigen::Matrix3f mx, my, mz;
    mx = Eigen::AngleAxisf(xAngRAD, Eigen::Vector3f::UnitX());
    my = Eigen::AngleAxisf(yAngRAD, Eigen::Vector3f::UnitY());
    mz = Eigen::AngleAxisf(zAngRAD, Eigen::Vector3f::UnitZ());

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
