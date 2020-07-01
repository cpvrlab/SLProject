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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include <list>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <opencv2/core.hpp>
#include <WorkingSet.h>
#include <LocalMap.h>

class WAIKeyFrame;
class WAIMap;
class WAIKeyFrameDB;
class WAIMapPoint;

namespace ORB_SLAM2
{

//class Tracking;
class LoopClosing;

class LocalMapping
{
public:
    LocalMapping(WAIMap* pMap, const float bMonocular, WAIOrbVocabulary* vocabulary, float cullRedundantPerc = 0.9);
    void SetLoopCloser(LoopClosing* pLoopCloser);

    // Main function
    void Run();
    void LocalOptimize();
    void ProcessKeyFrames();

    //ghm1
    void RunOnce();
    void InsertKeyFrame(WAIKeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestContinue();
    void RequestReset();
    void Release();
    bool isStopped();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    //ghm1
    void reset();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue()
    {
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return (int)mlNewKeyFrames.size();
    }

    std::thread* AddLocalBAThread();

protected:

    bool CheckNewKeyFrames();
    WAIKeyFrame* GetNewKeyFrame();
    void ProcessNewKeyFrame(WAIKeyFrame * kf);
    void CreateNewMapPoints(WAIKeyFrame * kf);

    void MapPointCulling(WAIKeyFrame * kf);

    void searchNeihborsLocalMap(LocalMap &lmap, WorkingSet &ws, WAIKeyFrame* frame);
    void SearchInNeighbors(LocalMap &lmap);
    void SearchInNeighbors(WAIKeyFrame * kf);

    void KeyFrameCulling(WAIKeyFrame* frame, WorkingSet &ws);
    void KeyFrameCulling(WAIKeyFrame* frame);

    cv::Mat ComputeF12(WAIKeyFrame*& pKF1, WAIKeyFrame*& pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat& v);

    WorkingSet workingSet;

    WAIMap* mpMap;
    bool    mbMonocular;

    void       ResetIfRequested();
    bool       mbResetRequested;
    std::mutex mMutexReset;

    bool       CheckPause();
    bool       mbPaused;

    bool       CheckFinish();
    void       SetFinish();
    bool       mbFinishRequested;
    bool       mbFinished;
    int        mFinishedBA;
    int        mMappingThreads;
    std::mutex mMutexFinish;

    std::queue<WAIKeyFrame*> toLocalAdjustment;
    std::mutex mMutexMapping;
    std::mutex mStateMutex;

    std::mutex mMutexPause;
    bool       mPause;

    LoopClosing* mpLoopCloser;

    std::mutex mMutexNewKFs;
    bool       mbAcceptKeyFrames;
    std::list<WAIKeyFrame*> mlNewKeyFrames;
    std::list<WAIMapPoint*> mlpRecentAddedMapPoints;


    bool       mbAbortBA;
    std::mutex mMutexAccept;

    WAIOrbVocabulary* _vocabulary = NULL;

    // A keyframe is considered redundant if the _cullRedundantPerc of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    const float _cullRedundantPerc;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
