#include <WAISlam.h>
#include <AverageTiming.h>

#define MIN_FRAMES 0
#define MAX_FRAMES 30

void WAISlamTools::drawKeyPointInfo(WAIFrame& frame, cv::Mat& image)
{
    //show rectangle for all keypoints in current image
    for (size_t i = 0; i < frame.N; i++)
    {
        //Use distorted points because we have to undistort the image later
        //const auto& pt = mCurrentFrame.mvKeys[i].pt;
        const auto& pt = frame.mvKeys[i].pt;
        cv::rectangle(image,
                      cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                      cv::Scalar(0, 0, 255));
    }
}

void WAISlamTools::drawKeyPointMatches(WAIFrame& frame, cv::Mat& image)
{
    for (size_t i = 0; i < frame.N; i++)
    {
        if (frame.mvpMapPoints[i])
        {
            if (!frame.mvbOutlier[i])
            {
                if (frame.mvpMapPoints[i]->Observations() > 0)
                {
                    //Use distorted points because we have to undistort the image later
                    const auto& pt = frame.mvKeys[i].pt;
                    cv::rectangle(image,
                                  cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                                  cv::Scalar(0, 255, 0));
                }
            }
        }
    }
}

void WAISlamTools::drawInitInfo(InitializerData& iniData, WAIFrame& newFrame, cv::Mat& imageRGB)
{
    for (unsigned int i = 0; i < iniData.initialFrame.mvKeys.size(); i++)
    {
        cv::rectangle(imageRGB,
                      iniData.initialFrame.mvKeys[i].pt,
                      cv::Point(iniData.initialFrame.mvKeys[i].pt.x + 3, iniData.initialFrame.mvKeys[i].pt.y + 3),
                      cv::Scalar(0, 0, 255));
    }

    for (unsigned int i = 0; i < iniData.iniMatches.size(); i++)
    {
        if (iniData.iniMatches[i] >= 0)
        {
            cv::line(imageRGB,
                     iniData.initialFrame.mvKeys[i].pt,
                     newFrame.mvKeys[iniData.iniMatches[i]].pt,
                     cv::Scalar(0, 255, 0));
        }
    }
}

bool WAISlamTools::initialize(InitializerData& iniData,
                              WAIFrame&        frame,
                              ORBVocabulary*   voc,
                              LocalMap&        localMap,
                              int              mapPointsNeeded,
                              WAIKeyFrameDB*   keyFrameDatabase)
{
    int matchesNeeded = 100;

    if (!iniData.initializer)
    {
        // Set Reference Frame
        if (frame.mvKeys.size() > matchesNeeded)
        {
            iniData.initialFrame = WAIFrame(frame);
            iniData.prevMatched.resize(frame.mvKeysUn.size());
            //ghm1: we store the undistorted keypoints of the initial frame in an extra vector
            //todo: why not using mInitialFrame.mvKeysUn????
            for (size_t i = 0; i < frame.mvKeysUn.size(); i++)
                iniData.prevMatched[i] = frame.mvKeysUn[i].pt;

            iniData.initializer = new ORB_SLAM2::Initializer(frame, 1.0, 200);
            //ghm1: clear mvIniMatches. it contains the index of the matched keypoint in the current frame
            fill(iniData.iniMatches.begin(), iniData.iniMatches.end(), -1);
        }

        return false;
    }

    // Try to initialize
    if ((int)frame.mvKeys.size() <= matchesNeeded)
    {
        delete iniData.initializer;
        iniData.initializer = static_cast<Initializer*>(NULL);
        fill(iniData.iniMatches.begin(), iniData.iniMatches.end(), -1);
        return false;
    }

    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int        nmatches = matcher.SearchForInitialization(iniData.initialFrame, frame, iniData.prevMatched, iniData.iniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < matchesNeeded)
    {
        delete iniData.initializer;
        iniData.initializer = static_cast<Initializer*>(NULL);
        return false;
    }

    cv::Mat      Rcw;            // Current Camera Rotation
    cv::Mat      tcw;            // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

    if (iniData.initializer->Initialize(frame, iniData.iniMatches, Rcw, tcw, iniData.iniPoint3D, vbTriangulated))
    {
        for (size_t i = 0, iend = iniData.iniMatches.size(); i < iend; i++)
        {
            if (iniData.iniMatches[i] >= 0 && !vbTriangulated[i])
            {
                iniData.iniMatches[i] = -1;
                nmatches--;
            }
        }

        // Set Frame Poses
        iniData.initialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(Tcw.rowRange(0, 3).col(3));
        frame.SetPose(Tcw);

        //ghm1: reset nNextId to 0! This is important otherwise the first keyframe cannot be identified via its id and a lot of stuff gets messed up!
        //One problem we identified is in UpdateConnections: the first one is not allowed to have a parent,
        //because the second will set the first as a parent too. We get problems later during culling.
        //This also fixes a problem in first GlobalBundleAdjustment which messed up the map after a reset.
        WAIKeyFrame::nNextId = 0;

        // Create KeyFrames
        WAIKeyFrame* pKFini = new WAIKeyFrame(iniData.initialFrame, keyFrameDatabase);
        WAIKeyFrame* pKFcur = new WAIKeyFrame(frame, keyFrameDatabase);

        pKFini->ComputeBoW(voc);
        pKFcur->ComputeBoW(voc);

        // Create MapPoints and associate to keyframes
        for (size_t i = 0; i < iniData.iniMatches.size(); i++)
        {
            if (iniData.iniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(iniData.iniPoint3D[i]);

            WAIMapPoint* pMP = new WAIMapPoint(worldPos, pKFcur);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, iniData.iniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, iniData.iniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            frame.mvpMapPoints[iniData.iniMatches[i]] = pMP;
            frame.mvbOutlier[iniData.iniMatches[i]]   = false;
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Set median depth to 1
        float medianDepth    = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < mapPointsNeeded)
        {
            WAI_LOG("Wrong initialization, reseting...");
            keyFrameDatabase->clear();
            WAIKeyFrame::nNextId            = 0;
            WAIFrame::nNextId               = 0;
            WAIFrame::mbInitialComputations = true;
            WAIMapPoint::nNextId            = 0;

            return false;
        }

        localMap.keyFrames.push_back(pKFcur);
        localMap.keyFrames.push_back(pKFini);
        // Scale initial baseline
        cv::Mat Tc2w               = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<WAIMapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                WAIMapPoint* pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
                localMap.mapPoints.push_back(pMP);
            }
        }

        frame.SetPose(pKFcur->GetPose());
        localMap.lastKF     = pKFcur;
        localMap.refKF      = pKFcur;
        frame.mpReferenceKF = pKFcur;

        return true;
    }
    return false;
}

bool WAISlamTools::genInitialMap(WAIMap*       map,
                                 LocalMapping* localMapper,
                                 LoopClosing*  loopCloser,
                                 LocalMap&     localMap,
                                 bool          serial)
{
    if (localMap.keyFrames.size() != 2)
    {
        return false;
    }

    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate, std::defer_lock);
    lock.lock();
    // Insert KFs in the map
    map->AddKeyFrame(localMap.keyFrames[0]);
    map->AddKeyFrame(localMap.keyFrames[1]);

    //Add to Map
    for (size_t i = 0; i < localMap.mapPoints.size(); i++)
    {
        WAIMapPoint* pMP = localMap.mapPoints[i];
        if (pMP)
            map->AddMapPoint(pMP);
    }

    // Bundle Adjustment
    Optimizer::GlobalBundleAdjustemnt(map, 20);

    localMapper->InsertKeyFrame(localMap.keyFrames[0]);
    localMapper->InsertKeyFrame(localMap.keyFrames[1]);

    map->SetReferenceMapPoints(localMap.mapPoints);
    map->mvpKeyFrameOrigins.push_back(localMap.keyFrames[0]);

    if (serial)
    {
        localMapper->RunOnce();
        localMapper->RunOnce();
    }

    return true;
}

bool WAISlamTools::tracking(WAIMap*        map,
                            WAIKeyFrameDB* keyFrameDatabase,
                            LocalMap&      localMap,
                            WAIFrame&      frame,
                            WAIFrame&      lastFrame,
                            int            lastRelocFrameId,
                            cv::Mat&       velocity,
                            int&           inliers)
{
    //std::unique_lock<std::mutex> lock(map->mMutexMapUpdate, std::defer_lock);
    //lock.lock();
    inliers = 0;

    if (!trackWithMotionModel(velocity, lastFrame, frame))
    {
        if (!trackReferenceKeyFrame(localMap, lastFrame, frame))
        {
            return false;
        }
    }

    return trackLocalMap(localMap, frame, lastRelocFrameId, inliers);
}

bool WAISlamTools::trackLocalMap(LocalMap& localMap,
                                 WAIFrame& frame,
                                 int       lastRelocFrameId,
                                 int&      inliers)
{
    updateLocalMap(frame, localMap);
    if (!localMap.keyFrames.empty())
    {
        frame.mpReferenceKF = localMap.refKF;
        inliers             = trackLocalMapPoints(localMap, lastRelocFrameId, frame);
        if (inliers > 50)
            return true;
    }
    return false;
}

void WAISlamTools::mapping(WAIMap*        map,
                           WAIKeyFrameDB* keyFrameDatabase,
                           LocalMap&      localMap,
                           LocalMapping*  localMapper,
                           WAIFrame&      frame,
                           int            inliers)
{
    if (needNewKeyFrame(map, localMap, localMapper, frame, inliers))
    {
        createNewKeyFrame(localMapper, localMap, map, keyFrameDatabase, frame);
        //TODO: test if should be here of outside the if statement
        for (int i = 0; i < frame.N; i++)
        {
            if (frame.mvpMapPoints[i] && frame.mvbOutlier[i])
            {
                frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
            }
        }
    }
}

void WAISlamTools::serialMapping(WAIMap*        map,
                                 WAIKeyFrameDB* keyFrameDatabase,
                                 LocalMap&      localMap,
                                 LocalMapping*  localMapper,
                                 LoopClosing*   loopCloser,
                                 WAIFrame&      frame,
                                 int            inliers)
{
    if (needNewKeyFrame(map, localMap, localMapper, frame, inliers))
    {
        createNewKeyFrame(localMapper, localMap, map, keyFrameDatabase, frame);
        //TODO: test if should be here of outside the if statement
        for (int i = 0; i < frame.N; i++)
        {
            if (frame.mvpMapPoints[i] && frame.mvbOutlier[i])
            {
                frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
            }
        }

        localMapper->RunOnce();
        loopCloser->RunOnce();
    }
}

void WAISlamTools::motionModel(WAIFrame& frame,
                               WAIFrame& lastFrame,
                               cv::Mat&  velocity,
                               cv::Mat&  pose)
{
    if (!lastFrame.mTcw.empty())
    {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        lastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3)); //mRwc

        //this is the translation of the frame w.r.t the world
        const auto& cc = lastFrame.GetCameraCenter();

        cc.copyTo(LastTwc.rowRange(0, 3).col(3));

        velocity = frame.mTcw * LastTwc;
        pose     = frame.mTcw.clone();
    }
    else
    {
        velocity = cv::Mat();
    }
}

void WAISlamTools::createNewKeyFrame(LocalMapping*  localMapper,
                                     LocalMap&      lmap,
                                     WAIMap*        map,
                                     WAIKeyFrameDB* keyFrameDatabase,
                                     WAIFrame&      frame)
{
    if (!localMapper->SetNotStop(true))
        return;

    WAIKeyFrame* pKF = new WAIKeyFrame(frame, keyFrameDatabase);

    lmap.lastKF         = pKF;
    lmap.refKF          = pKF;
    frame.mpReferenceKF = pKF;

    localMapper->InsertKeyFrame(pKF);
    localMapper->SetNotStop(false);
}

bool WAISlamTools::needNewKeyFrame(WAIMap*       map,
                                   LocalMap&     localMap,
                                   LocalMapping* localMapper,
                                   WAIFrame&     frame,
                                   int           nInliners)
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (localMapper->isStopped() || localMapper->stopRequested())
        return false;

    const int nKFs = map->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (frame.mnId < localMap.lastKF->mnId + MAX_FRAMES && nKFs > MAX_FRAMES)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    int nRefMatches = localMap.refKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = localMapper->AcceptKeyFrames();

    // Thresholds
    float thRefRatio = 0.9f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = frame.mnId >= localMap.lastKF->mnId + MAX_FRAMES;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (frame.mnId >= localMap.lastKF->mnId + MIN_FRAMES && bLocalMappingIdle);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((nInliners < nRefMatches * thRefRatio) && nInliners > 15);

    if ((c1a || c1b) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            localMapper->InterruptBA();
            return false;
        }
    }
    else
    {
        return false;
    }
}

bool WAISlamTools::relocalization(WAIFrame&      currentFrame,
                                  WAIMap*        waiMap,
                                  WAIKeyFrameDB* keyFrameDatabase,
                                  LocalMap&      localMap,
                                  int&           inliers)
{
    AVERAGE_TIMING_START("relocalization");
    // Compute Bag of Words Vector
    currentFrame.ComputeBoW();
    // Relocalization is performed when tracking is lost
    // Track Lost: Query WAIKeyFrame Database for keyframe candidates for relocalisation
    vector<WAIKeyFrame*> vpCandidateKFs;
    vpCandidateKFs = keyFrameDatabase->DetectRelocalizationCandidates(&currentFrame, true); //put boolean to argument

    //std::cout << "N after DetectRelocalizationCandidates: " << vpCandidateKFs.size() << std::endl;

    if (vpCandidateKFs.empty())
    {
        AVERAGE_TIMING_STOP("relocalization");
        return false;
    }

    //vector<WAIKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->keyFrames();
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    // Best match < 0.75 * second best match (default is 0.6)
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<WAIMapPoint*>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        WAIKeyFrame* pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, currentFrame, vvpMapPointMatches[i]);
            //cout << "Num matches: " << nmatches << endl;
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(currentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool       bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int          nInliers;
            bool         bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat    Tcw     = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(currentFrame.mTcw);

                set<WAIMapPoint*> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        currentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        currentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&currentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < currentFrame.N; io++)
                    if (currentFrame.mvbOutlier[io])
                        currentFrame.mvpMapPoints[io] = static_cast<WAIMapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again:
                //ghm1: mappoints seen in the keyframe which was found as candidate via BoW-search are projected into
                //the current frame using the position that was calculated using the matches from BoW matcher
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(currentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&currentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < currentFrame.N; ip++)
                                if (currentFrame.mvpMapPoints[ip])
                                    sFound.insert(currentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(currentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&currentFrame);

                                for (int io = 0; io < currentFrame.N; io++)
                                    if (currentFrame.mvbOutlier[io])
                                        currentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = trackLocalMap(localMap, currentFrame, currentFrame.mnId, inliers);
                    break;
                }
            }
        }
    }

    AVERAGE_TIMING_STOP("relocalization");
    return bMatch;
}

bool WAISlamTools::trackReferenceKeyFrame(LocalMap& map, WAIFrame& lastFrame, WAIFrame& frame)
{
    //This routine is called if current tracking state is OK but we have NO valid motion model
    //1. Berechnung des BoW-Vectors für den current frame
    //2. using BoW we search mappoint matches (from reference keyframe) with orb in current frame (ORB that belong to the same vocabulary node (at a certain level))
    //3. if there are less than 15 matches return.
    //4. we use the pose found for the last frame as initial pose for the current frame
    //5. This pose is optimized using the matches to map points found by BoW search with reference frame
    //6. Matches classified as outliers by the optimization routine are updated in the mvpMapPoints vector in the current frame and the valid matches are counted
    //7. If there are more than 10 valid matches the reference frame tracking was successful.

    AVERAGE_TIMING_START("trackReferenceKeyFrame");

    // Compute Bag of Words vector
    frame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher           matcher(0.7, true);
    vector<WAIMapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(map.refKF, frame, vpMapPointMatches);

    if (nmatches < 15)
    {
        AVERAGE_TIMING_STOP("trackReferenceKeyFrame");
        return false;
    }

    frame.mvpMapPoints = vpMapPointMatches;
    frame.SetPose(lastFrame.mTcw);

    Optimizer::PoseOptimization(&frame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < frame.N; i++)
    {
        if (frame.mvpMapPoints[i])
        {
            if (frame.mvbOutlier[i])
            {
                WAIMapPoint* pMP = frame.mvpMapPoints[i];

                frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
                frame.mvbOutlier[i]   = false;
                pMP->mbTrackInView    = false;
                pMP->mnLastFrameSeen  = frame.mnId;
                nmatches--;
            }
            else if (frame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }
    AVERAGE_TIMING_STOP("trackReferenceKeyFrame");
    return nmatchesMap >= 10;
}

int WAISlamTools::trackLocalMapPoints(LocalMap& localMap, int lastRelocFrameId, WAIFrame& frame)
{
    // Do not search map points already matched
    for (vector<WAIMapPoint*>::iterator vit = frame.mvpMapPoints.begin(), vend = frame.mvpMapPoints.end(); vit != vend; vit++)
    {
        WAIMapPoint* pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<WAIMapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = frame.mnId;
                pMP->mbTrackInView   = false;
            }
        }
    }

    int nToMatch = 0;

    // Project localmap points in frame and check its visibility
    for (vector<WAIMapPoint*>::iterator vit = localMap.mapPoints.begin(), vend = localMap.mapPoints.end(); vit != vend; vit++)
    {
        WAIMapPoint* pMP = *vit;
        if (pMP->mnLastFrameSeen == frame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        //TOTO(LULUC) add viewing angle parameter
        if (frame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int        th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if (frame.mnId - 1 == lastRelocFrameId)
            th = 5;
        matcher.SearchByProjection(frame, localMap.mapPoints, th);
    }

    // Optimize Pose
    Optimizer::PoseOptimization(&frame);
    int matchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < frame.N; i++)
    {
        if (frame.mvpMapPoints[i])
        {
            if (!frame.mvbOutlier[i])
            {
                frame.mvpMapPoints[i]->IncreaseFound();
                if (frame.mvpMapPoints[i]->Observations() > 0)
                {
                    matchesInliers++;
                }
                else
                {
                    frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
                }
            }
        }
    }

    //TOTO(LULUC) check why that is done after "matchLocalMapPoint" in modOrbSlam2
    //and why outlier flag is set to false if no observation
    /*
    for (int i = 0; i < frame.N; i++)
    {
        WAIMapPoint* pMP = frame.mvpMapPoints[i];
        if (pMP)
        {
            if (pMP->Observations() < 1)
            {
                frame.mvbOutlier[i]   = false;
                frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
            }
        }
    }
    */
    return matchesInliers;
}

void WAISlamTools::updateLocalMap(WAIFrame& frame, LocalMap& localMap)
{
    // Each map point vote for the keyframes in which it has been observed
    map<WAIKeyFrame*, int> keyframeCounter;
    for (int i = 0; i < frame.N; i++)
    {
        if (frame.mvpMapPoints[i])
        {
            WAIMapPoint* pMP = frame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<WAIKeyFrame*, size_t> observations = pMP->GetObservations();
                for (map<WAIKeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                frame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max        = 0;
    localMap.refKF = static_cast<WAIKeyFrame*>(NULL);

    localMap.keyFrames.clear();
    localMap.keyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (auto it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        WAIKeyFrame* pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max) //Should be computed with parent and child added to localocalMap
        {
            max            = it->second;
            localMap.refKF = pKF;
        }

        localMap.keyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = frame.mnId; // <==== UHHH WHAT IS THAT???
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (auto itKF = localMap.keyFrames.begin(); itKF != localMap.keyFrames.end(); itKF++)
    {
        // Limit the number of keyframes
        if (localMap.keyFrames.size() > 80)
            break;

        WAIKeyFrame* pKF = *itKF;

        const vector<WAIKeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<WAIKeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            WAIKeyFrame* pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != frame.mnId) //to ensure not already added at previous step
                {
                    localMap.keyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = frame.mnId;
                    break;
                }
            }
        }

        const set<WAIKeyFrame*> spChilds = pKF->GetChilds();
        for (set<WAIKeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            WAIKeyFrame* pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != frame.mnId)
                {
                    localMap.keyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = frame.mnId;
                    break;
                }
            }
        }

        WAIKeyFrame* pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != frame.mnId)
            {
                localMap.keyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = frame.mnId;
                break;
            }
        }
    }

    localMap.mapPoints.clear();
    for (vector<WAIKeyFrame*>::const_iterator itKF = localMap.keyFrames.begin(), itEndKF = localMap.keyFrames.end(); itKF != itEndKF; itKF++)
    {
        WAIKeyFrame*               pKF   = *itKF;
        const vector<WAIMapPoint*> vpMPs = pKF->GetMapPointMatches();

        for (vector<WAIMapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            WAIMapPoint* pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == frame.mnId)
                continue;
            if (!pMP->isBad())
            {
                localMap.mapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = frame.mnId;
            }
        }
    }
}

bool WAISlamTools::trackWithMotionModel(cv::Mat velocity, WAIFrame& previousFrame, WAIFrame& frame)
{
    AVERAGE_TIMING_START("trackWithMotionModel");

    if (velocity.empty() || frame.mnId > previousFrame.mnId + 1)
        return false;

    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    //WAIKeyFrame* pRef = previousFrame.mpReferenceKF;
    //previousFrame.SetPose(last.lastFrameRelativePose * pRef->GetPose());

    //this adds the motion differnce between the last and the before-last frame to the pose of the last frame to estimate the position of the current frame
    frame.SetPose(velocity * previousFrame.mTcw);

    fill(frame.mvpMapPoints.begin(), frame.mvpMapPoints.end(), static_cast<WAIMapPoint*>(NULL));

    // Project points seen in previous frame
    int th       = 15;
    int nmatches = matcher.SearchByProjection(frame, previousFrame, th, true);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(frame.mvpMapPoints.begin(), frame.mvpMapPoints.end(), static_cast<WAIMapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(frame, previousFrame, 2 * th, true);
    }

    if (nmatches < 20)
    {
        AVERAGE_TIMING_STOP("trackWithMotionModel");
        return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&frame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < frame.N; i++)
    {
        if (frame.mvpMapPoints[i])
        {
            if (frame.mvbOutlier[i])
            {
                WAIMapPoint* pMP = frame.mvpMapPoints[i];

                frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
                frame.mvbOutlier[i]   = false;
                pMP->mbTrackInView    = false;
                pMP->mnLastFrameSeen  = frame.mnId;
                nmatches--;
            }
            else if (frame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    AVERAGE_TIMING_STOP("trackWithMotionModel");

    return nmatchesMap >= 10;
}

WAISlam::WAISlam(cv::Mat      intrinsic,
                 cv::Mat      distortion,
                 std::string  orbVocFile,
                 KPextractor* extractor)
{
    WAISlam(intrinsic, distortion, orbVocFile, extractor, false, false, false);
}

WAISlam::WAISlam(cv::Mat      intrinsic,
                 cv::Mat      distortion,
                 std::string  orbVocFile,
                 KPextractor* extractor,
                 bool         trackingOnly,
                 bool         serial,
                 bool         retainImg)
{
    _iniData.initializer = nullptr;
    _serial              = serial;
    _trackingOnly        = trackingOnly;
    _retainImg           = retainImg;

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIMapPoint::nNextId            = 0;
    WAIFrame::mbInitialComputations = true;

    _distortion      = distortion.clone();
    _cameraIntrinsic = intrinsic.clone();

    if (!WAIOrbVocabulary::initialize(orbVocFile))
        throw std::runtime_error("ModeOrbSlam2: could not find vocabulary file: " + orbVocFile);
    _voc = WAIOrbVocabulary::get();

    _keyFrameDatabase = new WAIKeyFrameDB(*_voc);
    _globalMap        = new WAIMap("Map");
    _extractor        = extractor;

    _localMapping = new ORB_SLAM2::LocalMapping(_globalMap, 1, _voc, 0.95);
    _loopClosing  = new ORB_SLAM2::LoopClosing(_globalMap, _keyFrameDatabase, _voc, false, false);

    _localMapping->SetLoopCloser(_loopClosing);
    _loopClosing->SetLocalMapper(_localMapping);

    if (!serial)
    {
        _localMappingThread = new std::thread(&LocalMapping::Run, _localMapping);
        _loopClosingThread  = new std::thread(&LoopClosing::Run, _loopClosing);
    }

    _state               = TrackingState_Initializing;
    _iniData.initializer = nullptr;
    _cameraExtrinsic     = cv::Mat::eye(4, 4, CV_32F);

    _lastFrame = WAIFrame();
}

void WAISlam::reset()
{
    if (!_serial)
    {
        _localMapping->RequestReset();
        _loopClosing->RequestReset();
    }
    else
    {
        _localMapping->reset();
        _loopClosing->reset();
    }

    _keyFrameDatabase->clear();
    _globalMap->clear();
    _localMap.keyFrames.clear();
    _localMap.mapPoints.clear();
    _localMap.refKF = nullptr;

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;
    _state                          = TrackingState_Initializing;
}

bool WAISlam::update(cv::Mat& imageGray)
{
    WAIFrame                     frame = WAIFrame(imageGray, 0.0, _extractor, _cameraIntrinsic, _distortion, _voc, _retainImg);
    std::unique_lock<std::mutex> guard(_mutexStates);

    switch (_state)
    {
        case TrackingState_Initializing: {
            if (initialize(_iniData, frame, _voc, _localMap, 100, _keyFrameDatabase))
            {
                if (genInitialMap(_globalMap, _localMapping, _loopClosing, _localMap, _serial))
                {
                    _lastRelocId = 0;
                    _state       = TrackingState_TrackingOK;
                }
            }
        }
        break;
        case TrackingState_TrackingOK: {
            int inliers;
            if (tracking(_globalMap, _keyFrameDatabase, _localMap, frame, _lastFrame, _lastRelocId, _velocity, inliers))
            {
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                if (_serial)
                    serialMapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, _loopClosing, frame, inliers);
                else
                    mapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, frame, inliers);
            }
            else
                _state = TrackingState_TrackingLost;
        }
        break;
        case TrackingState_TrackingLost: {
            int inliers;
            if (relocalization(frame, _globalMap, _keyFrameDatabase, _localMap, inliers))
            {
                _lastRelocId = frame.mnId;
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                if (_serial)
                    serialMapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, _loopClosing, frame, inliers);
                else
                    mapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, frame, inliers);

                _state = TrackingState_TrackingOK;
            }
        }
        break;
    }

    _lastFrame = WAIFrame(frame);
    return (_state == TrackingState_TrackingOK);
}

void WAISlam::drawInfo(cv::Mat& imageRGB,
                       bool     showInitLine,
                       bool     showKeyPoints,
                       bool     showKeyPointsMatched)
{
    if (_state == TrackingState_Initializing)
    {
        if (showInitLine)
            drawInitInfo(_iniData, _lastFrame, imageRGB);
    }
    else if (_state == TrackingState_TrackingOK)
    {
        if (showKeyPoints)
            drawKeyPointInfo(_lastFrame, imageRGB);
        if (showKeyPointsMatched)
            drawKeyPointMatches(_lastFrame, imageRGB);
    }
}

std::vector<WAIMapPoint*> WAISlam::getMatchedMapPoints(WAIFrame* frame)
{
    std::vector<WAIMapPoint*> result;

    for (int i = 0; i < frame->N; i++)
    {
        if (frame->mvpMapPoints[i])
        {
            if (!frame->mvbOutlier[i])
            {
                if (frame->mvpMapPoints[i]->Observations() > 0)
                    result.push_back(frame->mvpMapPoints[i]);
            }
        }
    }

    return result;
}

std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> WAISlam::getMatchedCorrespondances(WAIFrame* frame)
{
    std::vector<cv::Vec3f> points3d;
    std::vector<cv::Vec2f> points2d;

    for (int i = 0; i < frame->N; i++)
    {
        WAIMapPoint* mp = frame->mvpMapPoints[i];
        if (mp)
        {
            if (!frame->mvbOutlier[i])
            {
                if (mp->Observations() > 0)
                {
                    WAI::V3   _v = mp->worldPosVec();
                    cv::Vec3f v;
                    v[0] = _v.x;
                    v[1] = _v.y;
                    v[2] = _v.z;
                    points3d.push_back(v);
                    points2d.push_back(frame->mvKeysUn[i].pt);
                }
            }
        }
    }

    return std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>>(points3d, points2d);
}

void WAISlam::requestStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    _localMapping->RequestStop();
    while (!_localMapping->isStopped())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    _state = TrackingState_Idle;
}

bool WAISlam::hasStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return (_state == TrackingState_Idle);
}

bool WAISlam::retainImage()
{
    return false;
}

void WAISlam::resume()
{
    _localMapping->Release();
    _state = TrackingState_TrackingLost;
}

WAISlamMarker::WAISlamMarker(cv::Mat      intrinsic,
                             cv::Mat      distortion,
                             std::string  orbVocFile,
                             KPextractor* extractor,
                             KPextractor* markerExtractor,
                             std::string  markerFile,
                             bool         serial,
                             bool         retainImg)
  : WAISlam(intrinsic, distortion, orbVocFile, extractor, false, serial, retainImg)

{
    if (!markerFile.empty())
        _markerFrame = createMarkerFrame(markerFile, _markerExtractor);
}

WAIFrame WAISlamMarker::createMarkerFrame(std::string markerFile, KPextractor* markerExtractor)
{
    cv::Mat markerImgGray = cv::imread(markerFile, cv::IMREAD_GRAYSCALE);

    float fyCam = _cameraIntrinsic.at<float>(1, 1);
    float cyCam = _cameraIntrinsic.at<float>(1, 2);
    float fov   = 2.0f * atan2(cyCam, fyCam) * 180.0f / M_PI;

    float cx = (float)markerImgGray.cols * 0.5f;
    float cy = (float)markerImgGray.rows * 0.5f;
    float fy = cy / tanf(fov * 0.5f * M_PI / 180.0);
    float fx = fy;

    // TODO(dgj1): pass actual calibration for marker frame?
    cv::Mat markerCameraMat     = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat markerDistortionMat = cv::Mat::zeros(4, 1, CV_32F);

    WAIFrame result = WAIFrame(markerImgGray, 0.0f, markerExtractor, markerCameraMat, markerDistortionMat, _voc, true);
    result          = WAIFrame(markerImgGray, 0.0f, markerExtractor, markerCameraMat, markerDistortionMat, _voc, true);
    result          = WAIFrame(markerImgGray, 0.0f, markerExtractor, markerCameraMat, markerDistortionMat, _voc, true);
    return result;
}

bool WAISlamMarker::doMarkerMapPreprocessing(std::string markerFile,
                                             cv::Mat&    nodeTransform,
                                             float       markerWidthInM)
{
    // Additional steps to save marker map
    // 1. Find matches to marker on two keyframes
    // 1.a Extract features from marker image
    WAIFrame markerFrame = createMarkerFrame(markerFile, _markerExtractor);

    // 1.b Find keyframes with enough matches to marker image
    std::vector<WAIKeyFrame*> kfs = _globalMap->GetAllKeyFrames();

    WAIKeyFrame* matchedKf1 = nullptr;
    WAIKeyFrame* matchedKf2 = nullptr;

    cv::Mat ul = cv::Mat(cv::Point3f(0, 0, 1));
    cv::Mat ur = cv::Mat(cv::Point3f(markerFrame.imgGray.cols, 0, 1));
    cv::Mat ll = cv::Mat(cv::Point3f(0, markerFrame.imgGray.rows, 1));
    cv::Mat lr = cv::Mat(cv::Point3f(markerFrame.imgGray.cols, markerFrame.imgGray.rows, 1));

    cv::Mat ulKf1, urKf1, llKf1, lrKf1, ulKf2, urKf2, llKf2, lrKf2;
    cv::Mat ul3D, ur3D, ll3D, lr3D;
    cv::Mat AC, AB, n;

    for (int i1 = 0; i1 < kfs.size() - 1; i1++)
    {
        WAIKeyFrame* kfCand1 = kfs[i1];

        if (kfCand1->isBad()) continue;

        // 2. Calculate homography between the keyframes and marker
        cv::Mat homography1;
        if (findMarkerHomography(markerFrame, kfCand1, homography1, 50))
        {
            // 3.a Calculate position of the markers cornerpoints on first keyframe in 2D
            // NOTE(dgj1): assumption that intrinsic camera parameters are the same
            // TODO(dgj1): think about this assumption
            ulKf1 = homography1 * ul;
            ulKf1 /= ulKf1.at<float>(2, 0);
            urKf1 = homography1 * ur;
            urKf1 /= urKf1.at<float>(2, 0);
            llKf1 = homography1 * ll;
            llKf1 /= llKf1.at<float>(2, 0);
            lrKf1 = homography1 * lr;
            lrKf1 /= lrKf1.at<float>(2, 0);

            for (int i2 = i1 + 1; i2 < kfs.size(); i2++)
            {
                WAIKeyFrame* kfCand2 = kfs[i2];

                if (kfCand2->isBad()) continue;

                cv::Mat homography2;
                if (findMarkerHomography(markerFrame, kfCand2, homography2, 50))
                {
                    // 3.b Calculate position of the markers cornerpoints on second keyframe in 2D
                    // NOTE(dgj1): assumption that intrinsic camera parameters are the same
                    // TODO(dgj1): think about this assumption
                    ulKf2 = homography2 * ul;
                    ulKf2 /= ulKf2.at<float>(2, 0);
                    urKf2 = homography2 * ur;
                    urKf2 /= urKf2.at<float>(2, 0);
                    llKf2 = homography2 * ll;
                    llKf2 /= llKf2.at<float>(2, 0);
                    lrKf2 = homography2 * lr;
                    lrKf2 /= lrKf2.at<float>(2, 0);

                    // 4. Triangulate position of the markers cornerpoints
                    cv::Mat Rcw1 = kfCand1->GetRotation();
                    cv::Mat Rwc1 = Rcw1.t();
                    cv::Mat tcw1 = kfCand1->GetTranslation();
                    cv::Mat Tcw1(3, 4, CV_32F);
                    Rcw1.copyTo(Tcw1.colRange(0, 3));
                    tcw1.copyTo(Tcw1.col(3));

                    const float& fx1    = kfCand1->fx;
                    const float& fy1    = kfCand1->fy;
                    const float& cx1    = kfCand1->cx;
                    const float& cy1    = kfCand1->cy;
                    const float& invfx1 = kfCand1->invfx;
                    const float& invfy1 = kfCand1->invfy;

                    cv::Mat Rcw2 = kfCand2->GetRotation();
                    cv::Mat Rwc2 = Rcw2.t();
                    cv::Mat tcw2 = kfCand2->GetTranslation();
                    cv::Mat Tcw2(3, 4, CV_32F);
                    Rcw2.copyTo(Tcw2.colRange(0, 3));
                    tcw2.copyTo(Tcw2.col(3));

                    const float& fx2    = kfCand2->fx;
                    const float& fy2    = kfCand2->fy;
                    const float& cx2    = kfCand2->cx;
                    const float& cy2    = kfCand2->cy;
                    const float& invfx2 = kfCand2->invfx;
                    const float& invfy2 = kfCand2->invfy;

                    {
                        cv::Mat ul1 = (cv::Mat_<float>(3, 1) << (ulKf1.at<float>(0, 0) - cx1) * invfx1, (ulKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat ul2 = (cv::Mat_<float>(3, 1) << (ulKf2.at<float>(0, 0) - cx2) * invfx2, (ulKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = ul1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = ul1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = ul2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = ul2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        ul3D = vt.row(3).t();

                        if (ul3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            ul3D = ul3D.rowRange(0, 3) / ul3D.at<float>(3);
                        }
                    }

                    {
                        cv::Mat ur1 = (cv::Mat_<float>(3, 1) << (urKf1.at<float>(0, 0) - cx1) * invfx1, (urKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat ur2 = (cv::Mat_<float>(3, 1) << (urKf2.at<float>(0, 0) - cx2) * invfx2, (urKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = ur1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = ur1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = ur2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = ur2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        ur3D = vt.row(3).t();

                        if (ur3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            ur3D = ur3D.rowRange(0, 3) / ur3D.at<float>(3);
                        }
                    }

                    {
                        cv::Mat ll1 = (cv::Mat_<float>(3, 1) << (llKf1.at<float>(0, 0) - cx1) * invfx1, (llKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat ll2 = (cv::Mat_<float>(3, 1) << (llKf2.at<float>(0, 0) - cx2) * invfx2, (llKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = ll1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = ll1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = ll2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = ll2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        ll3D = vt.row(3).t();

                        if (ll3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            ll3D = ll3D.rowRange(0, 3) / ll3D.at<float>(3);
                        }
                    }

                    {
                        cv::Mat lr1 = (cv::Mat_<float>(3, 1) << (lrKf1.at<float>(0, 0) - cx1) * invfx1, (lrKf1.at<float>(1, 0) - cy1) * invfy1, 1.0);
                        cv::Mat lr2 = (cv::Mat_<float>(3, 1) << (lrKf2.at<float>(0, 0) - cx2) * invfx2, (lrKf2.at<float>(1, 0) - cy2) * invfy2, 1.0);

                        // Linear Triangulation Method
                        cv::Mat A(4, 4, CV_32F);
                        A.row(0) = lr1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = lr1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = lr2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = lr2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                        cv::Mat w, u, vt;
                        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                        lr3D = vt.row(3).t();

                        if (lr3D.at<float>(3) != 0)
                        {
                            // Euclidean coordinates
                            lr3D = lr3D.rowRange(0, 3) / lr3D.at<float>(3);
                        }
                    }

                    AC = ll3D - ul3D;
                    AB = ur3D - ul3D;

                    cv::Vec3f vAC = AC;
                    cv::Vec3f vAB = AB;

                    cv::Vec3f vn = vAB.cross(vAC);
                    n            = cv::Mat(vn);

                    cv::Mat   AD  = lr3D - ul3D;
                    cv::Vec3f vAD = AD;

                    float d = cv::norm(vn.dot(vAD)) / cv::norm(vn);
                    if (d < 0.01f)
                    {
                        matchedKf1 = kfCand1;
                        matchedKf2 = kfCand2;

                        break;
                    }
                }
            }
        }

        if (matchedKf2) break;
    }

    if (!matchedKf1 || !matchedKf2)
    {
        return false;
    }

    // 5. Cull mappoints outside of marker
    std::vector<WAIMapPoint*> mapPoints = _globalMap->GetAllMapPoints();

    cv::Mat system = cv::Mat::zeros(3, 3, CV_32F);
    AC.copyTo(system.rowRange(0, 3).col(0));
    AB.copyTo(system.rowRange(0, 3).col(1));
    n.copyTo(system.rowRange(0, 3).col(2));

    cv::Mat systemInv = system.inv();

    for (int i = 0; i < mapPoints.size(); i++)
    {
        WAIMapPoint* mp = mapPoints[i];

        if (mp->isBad()) continue;

        cv::Mat sol = systemInv * (mp->GetWorldPos() - ul3D);

        if (sol.at<float>(0, 0) < 0 || sol.at<float>(0, 0) > 1 ||
            sol.at<float>(1, 0) < 0 || sol.at<float>(1, 0) > 1 ||
            sol.at<float>(2, 0) < -0.1f || sol.at<float>(2, 0) > 0.1f)
        {
            mp->SetBadFlag();
        }
    }
#if 1
    for (int i = 0; i < kfs.size(); i++)
    {
        WAIKeyFrame* kf = kfs[i];

        if (kf->mnId == 0 || kf->isBad()) continue;

        int mpCount = 0;

        std::vector<WAIMapPoint*> mps = kf->GetMapPointMatches();
        for (int j = 0; j < mps.size(); j++)
        {
            WAIMapPoint* mp = mps[j];

            if (!mp || mp->isBad()) continue;

            mpCount++;
        }

        if (mpCount <= 0)
        {
            kf->SetBadFlag();
        }
    }

#else

    // Cull redundant keyframes
    float cullRedundantPerc = 0.95f; //TODO(dgj1): make parametrizable
    for (int i = 0; i < kfs.size(); i++)
    {
        WAIKeyFrame* kf = kfs[i];

        vector<WAIKeyFrame*> vpLocalKeyFrames = kf->GetVectorCovisibleKeyFrames();

        for (vector<WAIKeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
        {
            WAIKeyFrame* pKF = *vit;
            //do not cull the first keyframe
            if (pKF->mnId == 0)
                continue;
            //do not cull fixed keyframes
            if (pKF->isFixed())
                continue;

            const vector<WAIMapPoint*> vpMapPoints = pKF->GetMapPointMatches();

            const int thObs                  = 3;
            int       nRedundantObservations = 0;
            int       nMPs                   = 0;
            for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
            {
                WAIMapPoint* pMP = vpMapPoints[i];
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        nMPs++;
                        if (pMP->Observations() > thObs)
                        {
                            const int&                           scaleLevel   = pKF->mvKeysUn[i].octave;
                            const std::map<WAIKeyFrame*, size_t> observations = pMP->GetObservations();
                            int                                  nObs         = 0;
                            for (std::map<WAIKeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                            {
                                WAIKeyFrame* pKFi = mit->first;
                                if (pKFi == pKF)
                                    continue;
                                const int& scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                                if (scaleLeveli <= scaleLevel + 1)
                                {
                                    nObs++;
                                    if (nObs >= thObs)
                                    {
                                        break;
                                    }
                                }
                            }
                            if (nObs >= thObs)
                            {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            if (nMPs == 0 || nRedundantObservations > cullRedundantPerc * nMPs)
            {
                pKF->SetBadFlag();
            }
        }
    }
#endif
    cv::Mat systemNorm               = cv::Mat::zeros(3, 3, CV_32F);
    systemNorm.rowRange(0, 3).col(0) = system.rowRange(0, 3).col(1) / cv::norm(AB);
    systemNorm.rowRange(0, 3).col(1) = system.rowRange(0, 3).col(0) / cv::norm(AC);
    systemNorm.rowRange(0, 3).col(2) = system.rowRange(0, 3).col(2) / cv::norm(n);

    cv::Mat systemNormInv = systemNorm.inv();

    nodeTransform   = cv::Mat::eye(4, 4, CV_32F);
    cv::Mat ul3Dinv = -systemNormInv * ul3D;
    ul3Dinv.copyTo(nodeTransform.rowRange(0, 3).col(3));
    systemNormInv.copyTo(nodeTransform.rowRange(0, 3).colRange(0, 3));

    cv::Mat scaleMat         = cv::Mat::eye(4, 4, CV_32F);
    float   markerWidthInRef = cv::norm(ul3D - ur3D);
    float   scaleFactor      = markerWidthInM / markerWidthInRef;
    scaleMat.at<float>(0, 0) = scaleFactor;
    scaleMat.at<float>(1, 1) = scaleFactor;
    scaleMat.at<float>(2, 2) = scaleFactor;

    nodeTransform = scaleMat * nodeTransform;

    if (_mpUL)
    {
        delete _mpUL;
        _mpUL = nullptr;
    }
    if (_mpUR)
    {
        delete _mpUR;
        _mpUR = nullptr;
    }
    if (_mpLL)
    {
        delete _mpLL;
        _mpLL = nullptr;
    }
    if (_mpLR)
    {
        delete _mpLR;
        _mpLR = nullptr;
    }

    _mpUL = new WAIMapPoint(0, ul3D, false);
    _mpUR = new WAIMapPoint(0, ur3D, false);
    _mpLL = new WAIMapPoint(0, ll3D, false);
    _mpLR = new WAIMapPoint(0, lr3D, false);

    return true;
}

bool WAISlamMarker::findMarkerHomography(WAIFrame&    markerFrame,
                                         WAIKeyFrame* kfCand,
                                         cv::Mat&     homography,
                                         int          minMatches)
{
    bool result = false;

    ORBmatcher matcher(0.9, true);

    std::vector<int> markerMatchesToCurrentFrame;
    int              nmatches = matcher.SearchForMarkerMap(markerFrame, *kfCand, markerMatchesToCurrentFrame);

    if (nmatches > minMatches)
    {
        std::vector<cv::Point2f> markerPoints;
        std::vector<cv::Point2f> framePoints;

        for (int j = 0; j < markerMatchesToCurrentFrame.size(); j++)
        {
            if (markerMatchesToCurrentFrame[j] >= 0)
            {
                markerPoints.push_back(markerFrame.mvKeysUn[j].pt);
                framePoints.push_back(kfCand->mvKeysUn[markerMatchesToCurrentFrame[j]].pt);
            }
        }

        homography = cv::findHomography(markerPoints,
                                        framePoints,
                                        cv::RANSAC);

        if (!homography.empty())
        {
            homography.convertTo(homography, CV_32F);

            result = true;
        }
    }

    return result;
}

void WAISlamMarker::reset()
{
    if (!_serial)
    {
        _localMapping->RequestReset();
        _loopClosing->RequestReset();
    }
    else
    {
        _localMapping->reset();
        _loopClosing->reset();
    }

    _keyFrameDatabase->clear();
    _globalMap->clear();
    _localMap.keyFrames.clear();
    _localMap.mapPoints.clear();
    _localMap.refKF = nullptr;

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;
    _state                          = TrackingState_Initializing;
}

bool WAISlamMarker::update(cv::Mat& imageGray)
{
    WAIFrame                     frame;
    std::unique_lock<std::mutex> guard(_mutexStates);

    switch (_state)
    {
        case TrackingState_Initializing:
            frame = WAIFrame(imageGray, 0.0, _markerExtractor, _cameraIntrinsic, _distortion, _voc, _retainImg);
            if (initialize(_iniData, frame, _voc, _localMap, 100, _keyFrameDatabase))
            {
                if (genInitialMap(_globalMap, _localMapping, _loopClosing, _localMap, _serial))
                {
                    _lastRelocId = 0;
                    _state       = TrackingState_TrackingOK;
                }
            }
            break;
        case TrackingState_TrackingOK: {
            frame = WAIFrame(imageGray, 0.0, _extractor, _cameraIntrinsic, _distortion, _voc, _retainImg);
            int inliers;
            if (tracking(_globalMap, _keyFrameDatabase, _localMap, frame, _lastFrame, _lastRelocId, _velocity, inliers))
            {
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                if (_serial)
                    serialMapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, _loopClosing, frame, inliers);
                else
                    mapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, frame, inliers);
            }
            else
                _state = TrackingState_TrackingLost;
        }
        break;
        case TrackingState_TrackingLost:
            int inliers;
            if (relocalization(frame, _globalMap, _keyFrameDatabase, _localMap, inliers))
            {
                _lastRelocId = frame.mnId;
                motionModel(frame, _lastFrame, _velocity, _cameraExtrinsic);
                if (_serial)
                    serialMapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, _loopClosing, frame, inliers);
                else
                    mapping(_globalMap, _keyFrameDatabase, _localMap, _localMapping, frame, inliers);

                _state = TrackingState_TrackingOK;
            }
            break;
    }

    _lastFrame = WAIFrame(frame);
    return (_state == TrackingState_TrackingOK);
}

std::vector<WAIMapPoint*> WAISlamMarker::getMarkerCornerMapPoints()
{
    std::vector<WAIMapPoint*> result;

    if (_mpUL)
    {
        if (_mpUL->isBad())
        {
            WAI_LOG("_mpUL->isBad()!!!");
        }
        result.push_back(_mpUL);
    }
    if (_mpUR)
    {
        if (_mpUR->isBad())
        {
            WAI_LOG("_mpUR->isBad()!!!");
        }
        result.push_back(_mpUR);
    }
    if (_mpLL)
    {
        if (_mpLL->isBad())
        {
            WAI_LOG("_mpLL->isBad()!!!");
        }
        result.push_back(_mpLL);
    }
    if (_mpLR)
    {
        if (_mpLR->isBad())
        {
            WAI_LOG("_mpLR->isBad()!!!");
        }
        result.push_back(_mpLR);
    }

    return result;
}

void WAISlamMarker::requestStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    _localMapping->RequestStop();
    while (!_localMapping->isStopped())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    _state = TrackingState_Idle;
}

bool WAISlamMarker::hasStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);
    return (_state == TrackingState_Idle);
}

bool WAISlamMarker::retainImage()
{
    return false;
}

void WAISlamMarker::resume()
{
    _localMapping->Release();
    _state = TrackingState_TrackingLost;
}
