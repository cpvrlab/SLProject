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

#include <OrbSlam/LoopClosing.h>
#include <OrbSlam/Sim3Solver.h>
#include <OrbSlam/Converter.h>
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/ORBmatcher.h>

#include <WAIKeyFrameDB.h>
#include <mutex>
#include <thread>

namespace ORB_SLAM2
{

LoopClosing::LoopClosing(WAIMap*        pMap,
                         ORBVocabulary* pVoc,
                         const bool     bFixScale,
                         const bool     manualLoopClose)
  : mbResetRequested(false),
    mbFinishRequested(false),
    mbFinished(true),
    mpMap(pMap),
    mpORBVocabulary(pVoc),
    mpMatchedKF(NULL),
    mLastLoopKFid(0),
    mbRunningGBA(false),
    mbFinishedGBA(true),
    mbStopGBA(false),
    mpThreadGBA(NULL),
    mbFixScale(bFixScale),
    mnFullBAIdx(0),
    _attemptLoopClose(!manualLoopClose),
    _manualLoopClose(manualLoopClose)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetLocalMapper(LocalMapping* pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void LoopClosing::SetVocabulary(ORBVocabulary* voc)
{
    mpORBVocabulary = voc;
}

void LoopClosing::Run()
{
    //mbFinished =false;

    while (1)
    {
        //Condition variable hints: we have three tasks in this loop:
        //process new keyframes, reset loopclosing, break while loop
        //All of these depend on own conditions. After one of these condition was changed, "loopContinue()" has to be called.
        {
            std::unique_lock<std::mutex> lock(_mutexLoop);
            _condVarLoop.wait(lock, [&] { return !_loopWait; });
        }
        //sleep again: if one participant is calling wake up in between the previous and the next call
        //the loop will be executed anyway!
        loopWait();

        if (CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            if (DetectLoop())
            {
                // Compute similarity transformation [sR|t]
                // In the stereo/RGBD case s=1
                if (ComputeSim3())
                {
                    // Perform loop fusion and pose graph optimization
                    CorrectLoop();
                    status(LOOP_CLOSE_STATUS_LOOP_CLOSED);

                    mpMap->incNumLoopClosings();

                    if (_manualLoopClose)
                    {
                        _attemptLoopClose = false;
                    }
                }
            }
        }

        ResetIfRequested();

        if (CheckFinish())
            break;
    }

    if (mpThreadGBA && mpThreadGBA->joinable())
        mpThreadGBA->join();

    SetFinish();
}

bool LoopClosing::RunOnce()
{
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames())
    {
        // Detect loop candidates and check covisibility consistency
        if (DetectLoop())
        {
            // Compute similarity transformation [sR|t]
            // In the stereo/RGBD case s=1
            if (ComputeSim3())
            {
                // Perform loop fusion and pose graph optimization
                doCorrectLoop();
                status(LOOP_CLOSE_STATUS_LOOP_CLOSED);

                mpMap->incNumLoopClosings();

                return true;
            }
        }
    }
    else
    {
        status(LOOP_CLOSE_STATUS_NO_NEW_KEYFRAME);
    }
    return false;
}

void LoopClosing::InsertKeyFrame(WAIKeyFrame* pKF)
{
    {
        std::lock_guard<std::mutex> lock(mMutexLoopQueue);
        if (pKF->mnId != 0)
        {
            mlpLoopKeyFrameQueue.push_back(pKF);
        }
    }
    loopContinue();
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return (!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    if (!shouldLoopCloseBeAttempted())
    {
        mpMap->GetKeyFrameDB()->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if (mpCurrentKF->mnId < mLastLoopKFid + 10)
    {
        status(LOOP_CLOSE_STATUS_NOT_ENOUGH_KEYFRAMES);
        mpMap->GetKeyFrameDB()->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<WAIKeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector&    CurrentBowVec        = mpCurrentKF->mBowVec;
    float                      minScore             = 1;
    for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++)
    {
        WAIKeyFrame* pKF = vpConnectedKeyFrames[i];
        if (pKF->isBad())
        {
            continue;
        }

        const DBoW2::BowVector& BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if (score < minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    int                  loopCandidateDetectionError = WAIKeyFrameDB::LOOP_DETECTION_ERROR_NONE;
    vector<WAIKeyFrame*> vpCandidateKFs              = mpMap->GetKeyFrameDB()->DetectLoopCandidates(mpCurrentKF, minScore, &loopCandidateDetectionError);
    {
        std::lock_guard<std::mutex> lock(mMutexNumCandidates);
        _numOfCandidates = vpCandidateKFs.size();
    }

    // If there are no loop candidates, just add new keyframe and return false
    if (vpCandidateKFs.empty())
    {
        switch (loopCandidateDetectionError)
        {
            case WAIKeyFrameDB::LOOP_DETECTION_ERROR_NO_CANDIDATES_WITH_COMMON_WORDS:
                status(LOOP_CLOSE_STATUS_NO_CANDIDATES_WITH_COMMON_WORDS);
                break;
            case WAIKeyFrameDB::LOOP_DETECTION_ERROR_NO_SIMILAR_CANDIDATES:
                status(LOOP_CLOSE_STATUS_NO_SIMILAR_CANDIDATES);
                break;
        }

        mpMap->GetKeyFrameDB()->add(mpCurrentKF);
        {
            std::lock_guard<std::mutex> lock(mMutexNumConsistentGroups);
            mvConsistentGroups.clear();
        }
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool>            vbConsistentGroup(mvConsistentGroups.size(), false);
    for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++)
    {
        WAIKeyFrame* pCandidateKF = vpCandidateKFs[i];

        set<WAIKeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent       = false;
        bool bConsistentForSomeGroup = false;
        for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
        {
            set<WAIKeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for (set<WAIKeyFrame*>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
            {
                if (sPreviousGroup.count(*sit))
                {
                    bConsistent             = true;
                    bConsistentForSomeGroup = true;
                    break;
                }
            }

            if (bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency  = nPreviousConsistency + 1;
                if (!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                }
                if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent = true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if (!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup, 0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    {
        std::lock_guard<std::mutex> lock(mMutexNumConsistentGroups);
        mvConsistentGroups = vCurrentConsistentGroups;
    }

    {
        std::lock_guard<std::mutex> lock(mMutexNumConsistentCandidates);
        _numOfConsistentCandidates = mvpEnoughConsistentCandidates.size();
    }

    // Add Current Keyframe to database
    mpMap->GetKeyFrameDB()->add(mpCurrentKF);

    if (mvpEnoughConsistentCandidates.empty())
    {
        status(LOOP_CLOSE_STATUS_NO_CONSISTENT_CANDIDATES);
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    // TODO(jan): we can never get here...
    mpCurrentKF->SetErase();
    return false;
}

bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75, true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<WAIMapPoint*>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates = 0; //candidates with enough matches

    for (int i = 0; i < nInitialCandidates; i++)
    {
        WAIKeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if (pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

        if (nmatches < 20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
            pSolver->SetRansacParameters(0.99, 20, 300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nInitialCandidates; i++)
        {
            if (vbDiscarded[i])
                continue;

            WAIKeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int          nInliers;
            bool         bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat     Scm     = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if (!Scm.empty())
            {
                vector<WAIMapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<WAIMapPoint*>(NULL));
                for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
                {
                    if (vbInliers[j])
                        vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                }

                cv::Mat     R = pSolver->GetEstimatedRotation();
                cv::Mat     t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

                g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                if (nInliers >= 20)
                {
                    bMatch      = true;
                    mpMatchedKF = pKF;
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
                    mg2oScw = gScm * gSmw;
                    mScw    = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        status(LOOP_CLOSE_STATUS_NO_OPTIMIZED_CANDIDATES);
        for (int i = 0; i < nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<WAIKeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for (vector<WAIKeyFrame*>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
    {
        WAIKeyFrame*         pKF         = *vit;
        vector<WAIMapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            WAIMapPoint* pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF = mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
    {
        if (mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if (nTotalMatches >= 40)
    {
        for (int i = 0; i < nInitialCandidates; i++)
            if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        status(LOOP_CLOSE_STATUS_NOT_ENOUGH_CONSISTENT_MATCHES);
        for (int i = 0; i < nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }
}

void LoopClosing::doCorrectLoop()
{
    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF] = mg2oScw;
    cv::Mat Twc                = mpCurrentKF->GetPoseInverse();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        for (vector<WAIKeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
        {
            WAIKeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if (pKFi != mpCurrentKF)
            {
                cv::Mat   Tic = Tiw * Twc;
                cv::Mat   Ric = Tic.rowRange(0, 3).colRange(0, 3);
                cv::Mat   tic = Tic.rowRange(0, 3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi] = g2oCorrectedSiw;
            }

            cv::Mat   Riw = Tiw.rowRange(0, 3).colRange(0, 3);
            cv::Mat   tiw = Tiw.rowRange(0, 3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi] = g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
        {
            WAIKeyFrame* pKFi            = mit->first;
            g2o::Sim3    g2oCorrectedSiw = mit->second;
            g2o::Sim3    g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

            vector<WAIMapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
            {
                WAIMapPoint* pMPi = vpMPsi[iMP];
                if (!pMPi)
                    continue;
                if (pMPi->isBad())
                    continue;
                if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat                     P3Dw             = pMPi->GetWorldPos();
                Eigen::Matrix<double, 3, 1> eigP3Dw          = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF      = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double          s    = g2oCorrectedSiw.scale();

            eigt *= (1. / s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
        {
            if (mvpCurrentMatchedPoints[i])
            {
                WAIMapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                WAIMapPoint* pCurMP  = mpCurrentKF->GetMapPoint(i);
                if (pCurMP)
                {
                    pCurMP->Replace(pLoopMP);
                    mpMap->EraseMapPoint(pCurMP);
                }
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP, i);
                    pLoopMP->AddObservation(mpCurrentKF, i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<WAIKeyFrame*, set<WAIKeyFrame*>> LoopConnections;

    for (vector<WAIKeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
    {
        WAIKeyFrame*         pKFi                = *vit;
        vector<WAIKeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        for (vector<WAIKeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for (vector<WAIKeyFrame*>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA  = true;
    mbFinishedGBA = false;
    mbStopGBA     = false;
    mpThreadGBA   = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosing::CorrectLoop()
{
    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if (mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    doCorrectLoop();
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose& CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
    {
        WAIKeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat   cvScw  = Converter::toCvMat(g2oScw);

        vector<WAIMapPoint*> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<WAIMapPoint*>(NULL));
        matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int          nLP = mvpLoopMapPoints.size();
        for (int i = 0; i < nLP; i++)
        {
            WAIMapPoint* pRep = vpReplacePoints[i];
            if (pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
                mpMap->EraseMapPoint(pRep);
            }
        }
    }
}

void LoopClosing::reset()
{
    mlpLoopKeyFrameQueue.clear();
    mLastLoopKFid    = 0;
    mbResetRequested = false;
}

void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }
    loopContinue();

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested)
    {
        reset();
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx = mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if (idx != mnFullBAIdx)
            return;

        if (!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<WAIKeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

            while (!lpKFtoCheck.empty())
            {
                WAIKeyFrame*            pKF     = lpKFtoCheck.front();
                const set<WAIKeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat                 Twc     = pKF->GetPoseInverse();
                for (set<WAIKeyFrame*>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++)
                {
                    WAIKeyFrame* pChild = *sit;
                    if (pChild->mnBAGlobalForKF != nLoopKF)
                    {
                        cv::Mat Tchildc         = pChild->GetPose() * Twc;
                        pChild->mTcwGBA         = Tchildc * pKF->mTcwGBA; //*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF = nLoopKF;
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<WAIMapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for (size_t i = 0; i < vpMPs.size(); i++)
            {
                WAIMapPoint* pMP = vpMPs[i];

                if (pMP->isBad())
                    continue;

                if (pMP->mnBAGlobalForKF == nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    WAIKeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if (pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                    cv::Mat Xc  = Rcw * pMP->GetWorldPos() + tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                    cv::Mat twc = Twc.rowRange(0, 3).col(3);

                    pMP->SetWorldPos(Rwc * Xc + twc);
                }
            }

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA  = false;
    }
}

void LoopClosing::RequestFinish()
{
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }
    loopContinue();
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

int LoopClosing::numOfCandidates()
{
    std::lock_guard<std::mutex> lock(mMutexNumCandidates);
    return _numOfCandidates;
}

int LoopClosing::numOfConsistentCandidates()
{
    std::lock_guard<std::mutex> lock(mMutexNumConsistentCandidates);
    return _numOfConsistentCandidates;
}

int LoopClosing::numOfConsistentGroups()
{
    std::lock_guard<std::mutex> lock(mMutexNumConsistentGroups);
    return mvConsistentGroups.size();
}

int LoopClosing::numOfKfsInQueue()
{
    std::lock_guard<std::mutex> lock(mMutexLoopQueue);
    return mlpLoopKeyFrameQueue.size();
}

void LoopClosing::status(LoopCloseStatus status)
{
    std::lock_guard<std::mutex> lock(mMutexStatus);
    _status = status;
}

const char* LoopClosing::getStatusString()
{
    switch (_status)
    {
        case LoopClosing::LOOP_CLOSE_STATUS_LOOP_CLOSED:
            return "loop closed";
        case LoopClosing::LOOP_CLOSE_STATUS_NOT_ENOUGH_CONSISTENT_MATCHES:
            return "not enough consistent matches";
        case LoopClosing::LOOP_CLOSE_STATUS_NOT_ENOUGH_KEYFRAMES:
            return "not enough keyframes";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_CONSISTENT_CANDIDATES:
            return "no consistent candidates";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_LOOP_CANDIDATES:
            return "no loop candidates";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_CANDIDATES_WITH_COMMON_WORDS:
            return "no candidates with common words";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_SIMILAR_CANDIDATES:
            return "no similar candidates";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_OPTIMIZED_CANDIDATES:
            return "no optimized candidates";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_NEW_KEYFRAME:
            return "no new keyframe";
        case LoopClosing::LOOP_CLOSE_STATUS_NONE:
        default:
            return "";
    }
}

bool LoopClosing::shouldLoopCloseBeAttempted()
{
    std::lock_guard<std::mutex> lock(mMutexLoopCloseAttempt);
    bool                        result = _attemptLoopClose;
    return result;
}

void LoopClosing::startLoopCloseAttempt()
{
    std::lock_guard<std::mutex> lock(mMutexLoopCloseAttempt);
    _attemptLoopClose = true;
}

void LoopClosing::loopContinue()
{
    {
        std::lock_guard<std::mutex> guard(_mutexLoop);
        _loopWait = false;
    }
    _condVarLoop.notify_one();
}
void LoopClosing::loopWait()
{
    std::lock_guard<std::mutex> guard(_mutexLoop);
    _loopWait = true;
}

} //namespace ORB_SLAM
