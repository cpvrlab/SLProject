//#############################################################################
//  File:      SLCVKeyframe.h
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


#ifndef SLCVKEYFRAME_H
#define SLCVKEYFRAME_H

#include <vector>
#include <SLCVCamera.h>
#include <DBoW2/DBoW2/BowVector.h>
#include <DBoW2/DBoW2/FeatureVector.h>
#include <OrbSlam/ORBVocabulary.h>
#include <SLGLTexture.h>
#include <opencv2/core/core.hpp>
#include <SLCVFrame.h>
#include <mutex>

using namespace ORB_SLAM2;

class SLCVMapPoint;
class SLCVKeyFrameDB;
class SLCVMap;

//-----------------------------------------------------------------------------
//! AR Keyframe node class
/*! A Keyframe is a camera with a position and additional information about key-
points that were found in this frame. It also contains descriptors for the found
keypoints.
*/
class SLCVKeyFrame
{
public:
    //!keyframe generation during map loading
    SLCVKeyFrame(const cv::Mat& Tcw, unsigned long id,
        float fx, float fy, float cx, float cy, size_t N, const std::vector<cv::KeyPoint>& vKeysUn, const cv::Mat& descriptors,
        ORBVocabulary* mpORBvocabulary, int nScaleLevels, float fScaleFactor, const std::vector<float>& vScaleFactors,
        const std::vector<float>& vLevelSigma2, const std::vector<float>& vInvLevelSigma2, int nMinX, int nMinY, int nMaxX, int nMaxY,
        const cv::Mat& K, SLCVKeyFrameDB* pKFDB, SLCVMap* pMap);
    //!keyframe generation from frame
    SLCVKeyFrame(SLCVFrame &F, SLCVMap* pMap, SLCVKeyFrameDB* pKFDB, bool retainImg = true);

    // Pose functions
    void SetPose(const SLCVMat& Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW(ORBVocabulary* orbVocabulary);

    // Covisibility graph functions
    void AddConnection(SLCVKeyFrame* pKF, int weight);
    void EraseConnection(SLCVKeyFrame* pKF);
    void UpdateConnections(bool buildSpanningTree = true);
    void UpdateBestCovisibles();
    std::set<SLCVKeyFrame*> GetConnectedKeyFrames();
    std::vector<SLCVKeyFrame* > GetVectorCovisibleKeyFrames();
    vector<SLCVKeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<SLCVKeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(SLCVKeyFrame* pKF);
    const std::map<SLCVKeyFrame*, int>& GetConnectedKfWeights();

    // Spanning tree functions
    void AddChild(SLCVKeyFrame* pKF);
    void EraseChild(SLCVKeyFrame* pKF);
    void ChangeParent(SLCVKeyFrame* pKF);
    std::set<SLCVKeyFrame*> GetChilds();
    SLCVKeyFrame* GetParent();
    bool hasChild(SLCVKeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(SLCVKeyFrame* pKF);
    std::set<SLCVKeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(SLCVMapPoint* pMP, size_t idx);
    void EraseMapPointMatch(SLCVMapPoint* pMP);
    void EraseMapPointMatch(const size_t &idx);
    void ReplaceMapPointMatch(const size_t &idx, SLCVMapPoint* pMP);
    std::set<SLCVMapPoint*> GetMapPoints();
    vector<SLCVMapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    SLCVMapPoint* GetMapPoint(const size_t &idx);
    bool hasMapPoint(SLCVMapPoint* mp);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp(int a, int b) {
        return a>b;
    }

    static bool lId(SLCVKeyFrame* pKF1, SLCVKeyFrame* pKF2) {
        return pKF1->mnId<pKF2->mnId;
    }

    //get estimated size of this object
    size_t getSizeOfCvMat(const cv::Mat& mat);
    size_t getSizeOf();

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame = 0;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery = 0;
    int mnLoopWords = 0;
    float mLoopScore = -1.0;
    long unsigned int mnRelocQuery = 0;
    int mnRelocWords = 0;
    float mRelocScore = -1.0f;

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy;/*, mbf, mb, mThDepth;*/

    // Number of KeyPoints
    const int N = 0;

    //undistorted keypoints
    const std::vector<cv::KeyPoint> mvKeysUn;

    //image feature descriptors
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;

    //original image
    cv::Mat imgGray;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    // SE3 Pose and camera center
    //! opencv coordinate representation: z-axis points to principlal point,
    //! x-axis to the right and y-axis down
    //! Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
    SLCVMat _Twc; //camera wrt world
    SLCVMat _Tcw; //world wrt camera
     //! camera center
    SLCVMat Ow;

    // MapPoints associated to keypoints (this array contains NULL for every
    //unassociated keypoint from original frame)
    std::vector<SLCVMapPoint*> mvpMapPoints;

    //pointer to keyframe database
    SLCVKeyFrameDB* _kfDb = NULL;

    // Grid over the image to speed up feature matching
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    std::map<SLCVKeyFrame*, int> mConnectedKeyFrameWeights;
    std::vector<SLCVKeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection = true;
    SLCVKeyFrame* mpParent = NULL;
    std::set<SLCVKeyFrame*> mspChildrens;
    std::set<SLCVKeyFrame*> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    SLCVMap* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;

public:
    //ghm1: added funtions
    //set path to texture image
    void setTexturePath(const string& path) { _pathToTexture = path; }
    const std::string& getTexturePath() { return _pathToTexture; }

    //! get visual representation as SLPoints
    SLMat4f getObjectMatrix();

private:
    //! this is a function from Frame, but we need it here for map loading
    void AssignFeaturesToGrid();
    //! this is a function from Frame, but we need it here for map loading
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    //path to background texture image
    string _pathToTexture;
};

#endif // !SLCVKEYFRAME_H
