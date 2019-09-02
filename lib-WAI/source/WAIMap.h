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

#ifndef WAIMAP_H
#define WAIMAP_H

#include <vector>
#include <string>
#include <mutex>
#include <set>

#include <opencv2/core.hpp>

#include <WAIMapPoint.h>
#include <WAIKeyFrameDB.h>
#include <WAIKeyFrame.h>
#include <WAIHelper.h>

using namespace std;

//-----------------------------------------------------------------------------
//!
/*! 
*/
class WAI_API WAIMap
{
    public:
    enum TransformType
    {
        ROT_X = 0,
        ROT_Y,
        ROT_Z,
        TRANS_X,
        TRANS_Y,
        TRANS_Z,
        SCALE
    };

    WAIMap(const string& name);
    ~WAIMap();

    void AddKeyFrame(WAIKeyFrame* pKF);
    void AddMapPoint(WAIMapPoint* pMP);
    void EraseMapPoint(WAIMapPoint* pMP);
    void EraseKeyFrame(WAIKeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<WAIMapPoint*>& vpMPs);
    void InformNewBigChange();
    int  GetLastBigChangeIdx();

    std::vector<WAIKeyFrame*> GetAllKeyFrames();
    std::vector<WAIMapPoint*> GetAllMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned int KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<WAIKeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    //transformation functions
    void    rotate(float degVal, int type);
    void    translate(float value, int type);
    void    scale(float value);
    void    applyTransformation(double value, TransformType type);
    cv::Mat buildTransMat(float& val, int type);
    cv::Mat buildRotMat(float& valDeg, int type);

    size_t getSizeOf();

    bool isKeyFrameInMap(WAIKeyFrame* pKF);

    void incNumLoopClosings();
    void setNumLoopClosings(int n);
    int  getNumLoopClosings();

    protected:
    std::set<WAIMapPoint*> mspMapPoints;
    std::set<WAIKeyFrame*> mspKeyFrames;

    std::vector<WAIMapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

    std::mutex _mutexLoopClosings;
    int        _numberOfLoopClosings = 0;
};

#endif // !WAIMAP_H
