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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "SLCVKeyFrame.h"
#include <OrbSlam/LocalMapping.h>
#include <SLCVMap.h>
#include <OrbSlam/ORBVocabulary.h>
//#include "Tracking.h"

#include <thread>
#include <mutex>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

class SLCVKeyFrameDB;

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;

class LoopClosing
{
public:

    typedef pair<set<SLCVKeyFrame*>,int> ConsistentGroup;    
    typedef map<SLCVKeyFrame*,g2o::Sim3,std::less<SLCVKeyFrame*>,
        Eigen::aligned_allocator<std::pair<const SLCVKeyFrame*, g2o::Sim3> > > KeyFrameAndPose;

public:

    LoopClosing(SLCVMap* pMap, SLCVKeyFrameDB* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();
    bool RunOnce();

    void InsertKeyFrame(SLCVKeyFrame *pKF);

    void RequestReset();
    void reset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    enum LoopCloseStatus
    {
        LOOP_CLOSE_STATUS_NONE,
        LOOP_CLOSE_STATUS_NOT_ENOUGH_KEYFRAMES,
        LOOP_CLOSE_STATUS_NO_LOOP_CANDIDATES,
        LOOP_CLOSE_STATUS_NO_CONSISTENT_CANDIDATES,
        LOOP_CLOSE_STATUS_NO_OPTIMIZED_CANDIDATES,
        LOOP_CLOSE_STATUS_NOT_ENOUGH_CONSISTENT_MATCHES,
        LOOP_CLOSE_STATUS_LOOP_CLOSED,
        LOOP_CLOSE_STATUS_NO_NEW_KEYFRAME
    };
    LoopCloseStatus getStatus() { return status; }

    int numOfLoopClosings();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();
    void doCorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    SLCVMap* mpMap;
    Tracking* mpTracker;

    SLCVKeyFrameDB* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<SLCVKeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    SLCVKeyFrame* mpCurrentKF;
    SLCVKeyFrame* mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<SLCVKeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<SLCVKeyFrame*> mvpCurrentConnectedKFs;
    std::vector<SLCVMapPoint*> mvpCurrentMatchedPoints;
    std::vector<SLCVMapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;

    bool mnFullBAIdx;

    LoopCloseStatus status = LOOP_CLOSE_STATUS_NONE;

    std::mutex mMutexNumLoopClosings;
    int _numLoopClosings;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
