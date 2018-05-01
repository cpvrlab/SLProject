//#############################################################################
//  File:      SLCVMap.cpp
//  Author:    Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

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

#include "stdafx.h"
#include "SLCVMap.h"
#include <SLMaterial.h>
#include <SLGLGenericProgram.h>
#include <SLPoints.h>
#include <SLCVKeyFrameDB.h>
#include <SLCVKeyFrame.h>
#include <SLCVMapNode.h>
#include <SLCVCalibration.h>

using namespace cv;

//-----------------------------------------------------------------------------
SLCVMap::SLCVMap(const string& name)
    : mnMaxKFid(0), mnBigChangeIdx(0),
    _mapNode(NULL)
{
}
//-----------------------------------------------------------------------------
SLCVMap::~SLCVMap()
{
    clear();
}
//-----------------------------------------------------------------------------
void SLCVMap::AddKeyFrame(SLCVKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if (pKF->mnId>mnMaxKFid)
        mnMaxKFid = pKF->mnId;
}
//-----------------------------------------------------------------------------
void SLCVMap::AddMapPoint(SLCVMapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}
//-----------------------------------------------------------------------------
void SLCVMap::EraseMapPoint(SLCVMapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}
//-----------------------------------------------------------------------------
void SLCVMap::EraseKeyFrame(SLCVKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}
//-----------------------------------------------------------------------------
void SLCVMap::SetReferenceMapPoints(const vector<SLCVMapPoint*> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}
//-----------------------------------------------------------------------------
void SLCVMap::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}
//-----------------------------------------------------------------------------
int SLCVMap::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}
//-----------------------------------------------------------------------------
std::vector<SLCVKeyFrame*> SLCVMap::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<SLCVKeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}
//-----------------------------------------------------------------------------
vector<SLCVMapPoint*> SLCVMap::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<SLCVMapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}
//-----------------------------------------------------------------------------
long unsigned int SLCVMap::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}
//-----------------------------------------------------------------------------
long unsigned int SLCVMap::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}
//-----------------------------------------------------------------------------
long unsigned int SLCVMap::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}
//-----------------------------------------------------------------------------
void SLCVMap::clear()
{
    //remove visual representation
    if (_mapNode)
        _mapNode->clearAll();

    for (auto* pt : mspMapPoints) {
        if (pt)
            delete pt;
    }
    for (auto* kf : mspKeyFrames) {
        if (kf)
            delete kf;
    }
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}
//-----------------------------------------------------------------------------
void SLCVMap::rotate(float value, int type)
{
    //transform to degree
    value *= SL_DEG2RAD;

    Mat rot = buildRotMat(value, type);
    cout << "rot: " << rot << endl;

    //rotate keyframes
    Mat Twc;
    for (auto& kf : mspKeyFrames)
    {
        //get and rotate
        Twc = kf->GetPose().inv();
        Twc = rot * Twc;
        //set back
        kf->SetPose(Twc.inv());
    }

    //rotate keypoints
    Mat Pw;
    Mat rot33 = rot.rowRange(0, 3).colRange(0, 3);
    for (auto& pt : mspMapPoints)
    {
        Pw = rot33 * pt->GetWorldPos();
        pt->SetWorldPos(rot33 * pt->GetWorldPos());
    }
}
//-----------------------------------------------------------------------------
void SLCVMap::translate(float value, int type)
{
    Mat trans = buildTransMat(value, type);

    cout << "trans: " << trans << endl;

    //rotate keyframes
    Mat Twc;
    for (auto& kf : mspKeyFrames)
    {
        //get and translate
        cv::Mat Twc = kf->GetPose().inv();
        Twc.rowRange(0, 3).col(3) += trans;
        //set back
        kf->SetPose(Twc.inv());
    }

    //rotate keypoints
    for (auto& pt : mspMapPoints)
    {
        pt->SetWorldPos(trans + pt->GetWorldPos());
    }
}
//-----------------------------------------------------------------------------
void SLCVMap::scale(float value)
{
    for (auto& kf : mspKeyFrames)
    {
        //get and translate
        cv::Mat Tcw = kf->GetPose();
        Tcw.rowRange(0, 3).col(3) *= value;
        //set back
        kf->SetPose(Tcw);
    }

    //rotate keypoints
    for (auto& pt : mspMapPoints)
    {
        pt->SetWorldPos(value * pt->GetWorldPos());
    }
}
//-----------------------------------------------------------------------------
void SLCVMap::applyTransformation(double value, TransformType type)
{
    std::lock_guard<std::mutex> guard(mMutexMap);

    //apply rotation, translation and scale to Keyframe and MapPoint poses
    cout << "apply transform with value: " << value << endl;
    switch (type)
    {
    case ROT_X:
        //build different transformation matrices for x,y and z rotation
        rotate((float)value, 0);
        break;
    case ROT_Y:
        rotate((float)value, 1);
        break;
    case ROT_Z:
        rotate((float)value, 2);
        break;
    case TRANS_X:
        translate((float)value, 0);
        break;
    case TRANS_Y:
        translate((float)value, 1);
        break;
    case TRANS_Z:
        translate((float)value, 2);
        break;
    case SCALE:
        scale((float)value);
        break;
    }

    //compute resulting values for map points
    for (auto& mp : mspMapPoints)
    {
        //mean viewing direction and depth
        mp->UpdateNormalAndDepth();
        mp->ComputeDistinctiveDescriptors();
    }

    //update scene objects
    //exchange all Keyframes (also change name)
    if (_mapNode)
        _mapNode->updateAll(*this);
    else
        SL_WARN_MSG("SLCVMap: applyTransformation: SLCVMapNode is NULL! Cannot update visualization!\n");
}
//-----------------------------------------------------------------------------
// Build rotation matrix
Mat SLCVMap::buildTransMat(float &val, int type)
{
    Mat trans = cv::Mat::zeros(3, 1, CV_32F);
    switch (type)
    {
    case 0:
        trans.at<float>(0, 0) = val;
        break;

    case 1:
        //!!turn sign of y coordinate
        trans.at<float>(1, 0) = -val;
        break;

    case 2:
        //!!turn sign of z coordinate
        trans.at<float>(2, 0) = -val;
        break;
    }

    return trans;
}
//-----------------------------------------------------------------------------
// Build rotation matrix
Mat SLCVMap::buildRotMat(float &valDeg, int type)
{
    Mat rot = Mat::ones(4, 4, CV_32F);

    switch (type)
    {
    case 0:
        // Calculate rotation about x axis
        rot = (Mat_<float>(4, 4) <<
            1, 0, 0, 0,
            0, cos(valDeg), -sin(valDeg), 0,
            0, sin(valDeg), cos(valDeg), 0,
            0, 0, 0, 1
            );
        break;

    case 1:
        // Calculate rotation about y axis
        rot = (Mat_<float>(4, 4) <<
            cos(valDeg), 0, sin(valDeg), 0,
            0, 1, 0, 0,
            -sin(valDeg), 0, cos(valDeg), 0,
            0, 0, 0, 1
            );
        //invert direction for Y
        rot = rot.inv();
        break;

    case 2:
        // Calculate rotation about z axis
        rot = (Mat_<float>(4, 4) <<
            cos(valDeg), -sin(valDeg), 0, 0,
            sin(valDeg), cos(valDeg), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
            );
        //invert direction for Z
        rot = rot.inv();
        break;
    }

    return rot;
}
//-----------------------------------------------------------------------------
void SLCVMap::setMapNode(SLCVMapNode* mapNode)
{
    _mapNode = mapNode;
}