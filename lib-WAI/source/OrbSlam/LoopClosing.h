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

#include <condition_variable>
#include <thread>
#include <mutex>

#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <OrbSlam/LocalMapping.h>
#include <OrbSlam/ORBVocabulary.h>

class WAIKeyFrameDB;
class WAIMap;
class WAIKeyFrame;

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;

class LoopClosing
{
    public:
    typedef pair<set<WAIKeyFrame*>, int>                                                                                              ConsistentGroup;
    typedef map<WAIKeyFrame*, g2o::Sim3, std::less<WAIKeyFrame*>, Eigen::aligned_allocator<std::pair<WAIKeyFrame* const, g2o::Sim3>>> KeyFrameAndPose;

    public:
    LoopClosing(WAIMap* pMap, WAIKeyFrameDB* pDB, ORBVocabulary* pVoc, const bool bFixScale, const bool manualLoopClose = false);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();
    bool RunOnce();

    void InsertKeyFrame(WAIKeyFrame* pKF);

    void RequestReset();
    void reset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA()
    {
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA()
    {
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }

    void RequestFinish();

    bool isFinished();

    enum LoopCloseStatus
    {
        LOOP_CLOSE_STATUS_NONE,
        LOOP_CLOSE_STATUS_NOT_ENOUGH_KEYFRAMES,
        LOOP_CLOSE_STATUS_NO_CANDIDATES_WITH_COMMON_WORDS,
        LOOP_CLOSE_STATUS_NO_SIMILAR_CANDIDATES,
        LOOP_CLOSE_STATUS_NO_LOOP_CANDIDATES,
        LOOP_CLOSE_STATUS_NO_CONSISTENT_CANDIDATES,
        LOOP_CLOSE_STATUS_NO_OPTIMIZED_CANDIDATES,
        LOOP_CLOSE_STATUS_NOT_ENOUGH_CONSISTENT_MATCHES,
        LOOP_CLOSE_STATUS_LOOP_CLOSED,
        LOOP_CLOSE_STATUS_NO_NEW_KEYFRAME
    };
    const char* getStatusString();

    int numOfCandidates();
    int numOfConsistentCandidates();
    int numOfConsistentGroups();
    int numOfKfsInQueue();

    void startLoopCloseAttempt();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose& CorrectedPosesMap);

    void CorrectLoop();
    void doCorrectLoop();

    void       ResetIfRequested();
    bool       mbResetRequested;
    std::mutex mMutexReset;

    bool       CheckFinish();
    void       SetFinish();
    bool       mbFinishRequested;
    bool       mbFinished;
    std::mutex mMutexFinish;

    WAIMap*   mpMap;
    Tracking* mpTracker;

    WAIKeyFrameDB* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping* mpLocalMapper;

    std::list<WAIKeyFrame*> mlpLoopKeyFrameQueue;
    std::mutex              mMutexLoopQueue;

    //replacement for thread sleep:
    std::mutex              _mutexLoop;
    std::condition_variable _condVarLoop;
    bool                    _loopWait = true;
    void                    loopContinue();
    void                    loopWait();

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    WAIKeyFrame*                 mpCurrentKF;
    WAIKeyFrame*                 mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<WAIKeyFrame*>    mvpEnoughConsistentCandidates;
    std::vector<WAIKeyFrame*>    mvpCurrentConnectedKFs;
    std::vector<WAIMapPoint*>    mvpCurrentMatchedPoints;
    std::vector<WAIMapPoint*>    mvpLoopMapPoints;
    cv::Mat                      mScw;
    g2o::Sim3                    mg2oScw;

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool         mbRunningGBA;
    bool         mbFinishedGBA;
    bool         mbStopGBA;
    std::mutex   mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;

    int mnFullBAIdx;

    LoopCloseStatus _status = LOOP_CLOSE_STATUS_NONE;
    std::mutex      mMutexStatus;
    void            status(LoopCloseStatus status);

    std::mutex mMutexLoopCloseAttempt;
    bool       shouldLoopCloseBeAttempted();
    bool       _attemptLoopClose;
    const bool _manualLoopClose;

    // Gui information
    std::mutex mMutexNumConsistentGroups;
    std::mutex mMutexNumCandidates;
    int        _numOfCandidates = 0;
    std::mutex mMutexNumConsistentCandidates;
    int        _numOfConsistentCandidates = 0;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
