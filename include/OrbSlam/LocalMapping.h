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

#include "SLCVKeyFrame.h"
#include "SLCVMap.h"
//#include "LoopClosing.h"
//#include "Tracking.h"
#include "SLCVKeyFrameDB.h"

//#include <mutex>

class SLCVMap;

namespace ORB_SLAM2
{


//class Tracking;
//class LoopClosing;

class LocalMapping
{
public:
    LocalMapping(SLCVMap* pMap, const float bMonocular, ORBVocabulary* mpORBvocabulary);

    //void SetLoopCloser(LoopClosing* pLoopCloser);

    //void SetTracker(Tracking* pTracker);

    // Main function
    //void Run();
    //ghm1
    void RunOnce();

    void InsertKeyFrame(SLCVKeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    //void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);
    //ghm1
    void reset();

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        //unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(SLCVKeyFrame* &pKF1, SLCVKeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    //std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    //std::mutex mMutexFinish;

    SLCVMap* mpMap;

    //LoopClosing* mpLoopCloser;
    //Tracking* mpTracker;

    std::list<SLCVKeyFrame*> mlNewKeyFrames;

    SLCVKeyFrame* mpCurrentKeyFrame;

    std::list<SLCVMapPoint*> mlpRecentAddedMapPoints;

    //std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    //std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    //std::mutex mMutexAccept;

    ORBVocabulary* mpORBvocabulary=NULL;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
