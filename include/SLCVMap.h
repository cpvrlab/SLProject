//#############################################################################
//  File:      SLCVMap.h
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

#ifndef SLCVMAP_H
#define SLCVMAP_H

#include <vector>
#include <string>
#include <mutex>
#include <SLCVMapPoint.h>

class SLPoints;
class SLCVKeyFrameDB;
class SLCVVKeyFrame;
class SLCVMapNode;

using namespace std;

//-----------------------------------------------------------------------------
//! 
/*! 
*/
class SLCVMap
{
public:
    enum TransformType {
        ROT_X = 0, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z, SCALE
    };

    SLCVMap(const string& name);
    ~SLCVMap();

    void AddKeyFrame(SLCVKeyFrame* pKF);
    void AddMapPoint(SLCVMapPoint *pMP);
    void EraseMapPoint(SLCVMapPoint *pMP);
    void EraseKeyFrame(SLCVKeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<SLCVMapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<SLCVKeyFrame*> GetAllKeyFrames();
    std::vector<SLCVMapPoint*> GetAllMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned int KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<SLCVKeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    //set map node for visu update
    void setMapNode(SLCVMapNode* mapNode);

    //transformation functions
    void rotate(float value, int type);
    void translate(float value, int type);
    void scale(float value);
    void applyTransformation(double value, TransformType type);
    cv::Mat buildTransMat(float &val, int type);
    cv::Mat buildRotMat(float &valDeg, int type);

protected:
    std::set<SLCVMapPoint*> mspMapPoints;
    std::set<SLCVKeyFrame*> mspKeyFrames;

    std::vector<SLCVMapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

    SLCVMapNode* _mapNode = NULL;
};

#endif // !SLCVMAP_H
