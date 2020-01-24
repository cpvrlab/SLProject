#include <LuluSlam.h>
#include <AverageTiming.h>


#define MIN_FRAMES  0
#define MAX_FRAMES  30


LuluSLAM::LuluSLAM(cv::Mat intrinsic,
                   cv::Mat distortion,
                   std::string orbVocFile,
                   KPextractor* extractorp)
{
    mIniData.initializer = nullptr;

    WAIKeyFrame::nNextId = 0;
    WAIFrame::nNextId    = 0;
    WAIMapPoint::nNextId = 0;
    WAIFrame::mbInitialComputations = true;

    mDistortion = distortion.clone();
    mCameraIntrinsic = intrinsic.clone();

    if (!WAIOrbVocabulary::initialize(orbVocFile))
        throw std::runtime_error("ModeOrbSlam2: could not find vocabulary file: " + orbVocFile);

    mVoc = WAIOrbVocabulary::get();
    mKeyFrameDatabase = new WAIKeyFrameDB(*mVoc);
    mGlobalMap = new WAIMap("Map");
    mExtractor = extractorp;
    mLocalMapping = new ORB_SLAM2::LocalMapping(mGlobalMap, 1, mVoc, 0.95);
    mLoopClosing = new ORB_SLAM2::LoopClosing(mGlobalMap, mKeyFrameDatabase, mVoc, false, false);

    mLocalMappingThread = new std::thread(&LocalMapping::Run, mLocalMapping);
    mLoopClosingThread = new std::thread(&LoopClosing::Run, mLoopClosing);

    mState = TrackingState_Initializing;
    mIniData.initializer = nullptr;
}

void LuluSLAM::drawInitInfo(initializerData& iniData, WAIFrame &newFrame, cv::Mat& imageRGB)
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

bool LuluSLAM::update(cv::Mat &imageGray, cv::Mat &imageRGB)
{
    WAIFrame frame = WAIFrame(imageGray, 0.0, mExtractor, mCameraIntrinsic, mDistortion, mVoc, false);
    mLastFrame = frame;

    std::unique_lock<std::mutex> guard(mMutexStates);

    switch(mState)
    {
        case TrackingState_Initializing:
            if (initialize(mIniData, mCameraIntrinsic, mDistortion, mVoc, mGlobalMap, mKeyFrameDatabase, mLmap, mLocalMapping, mLoopClosing, mLast, frame, mRelativeFramePoses))
            {
                mState = TrackingState_TrackingOK;
                mInitialized = true;
            }
            drawInitInfo(mIniData, frame, imageRGB);
            break;
        case TrackingState_TrackingOK:
            if (!trackingAndMapping(mCameraIntrinsic, mDistortion, mGlobalMap, mKeyFrameDatabase, mLast, mLmap, mLocalMapping, frame, mRelativeFramePoses, mCameraExtrinsic))
                mState = TrackingState_TrackingLost;

            drawKeyPointInfo(frame, imageRGB);
            drawKeyPointMatches(frame, imageRGB);
            break;
        case TrackingState_TrackingLost:
            if (relocalization(frame, mLast, mGlobalMap, mKeyFrameDatabase))
                mState = TrackingState_TrackingOK;
            break;
    }
    return (mState == TrackingState_TrackingOK);
}

cv::Mat LuluSLAM::getExtrinsic()
{
    return mCameraExtrinsic;
}

bool LuluSLAM::initialize(initializerData &iniData,
                          cv::Mat &camera,
                          cv::Mat &distortion,
                          ORBVocabulary *voc,
                          WAIMap *map,
                          WAIKeyFrameDB* keyFrameDatabase,
                          localMap &lmap,
                          LocalMapping *lmapper,
                          LoopClosing *loopClosing,
                          SLAMLatestState& last,
                          WAIFrame &frame,
                          list<cv::Mat> &relativeFramePoses)
{
    int matchesNeeded = 100;

    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate, std::defer_lock);
    lock.lock();

    if (!iniData.initializer)
    {
        // Set Reference Frame
        if (frame.mvKeys.size() > matchesNeeded)
        {
            iniData.initialFrame = WAIFrame(frame);
            WAIFrame mLastFrame = WAIFrame(frame);
            iniData.prevMatched.resize(frame.mvKeysUn.size());
            //ghm1: we store the undistorted keypoints of the initial frame in an extra vector
            //todo: why not using mInitialFrame.mvKeysUn????
            for (size_t i = 0; i < frame.mvKeysUn.size(); i++)
                iniData.prevMatched[i] = frame.mvKeysUn[i].pt;

            iniData.initializer = new ORB_SLAM2::Initializer(frame, 1.0, 200);
            //ghm1: clear mvIniMatches. it contains the index of the matched keypoint in the current frame
            fill(iniData.iniMatches.begin(), iniData.iniMatches.end(), -1);

            return false;
        }
    }
    else
    {
        // Try to initialize
        if ((int)frame.mvKeys.size() <= matchesNeeded)
        {
            delete iniData.initializer;
            iniData.initializer = static_cast<Initializer*>(NULL);
            fill(iniData.iniMatches.begin(), iniData.iniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int        nmatches = matcher.SearchForInitialization(iniData.initialFrame, frame, iniData.prevMatched, iniData.iniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches < matchesNeeded)
        {
            delete iniData.initializer;
            iniData.initializer = static_cast<Initializer*>(NULL);
            return;
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

            bool mapInitializedSuccessfully = createInitialMapMonocular(iniData, last, voc, map,
                                                                        lmapper,
                                                                        loopClosing,
                                                                        lmap,
                                                                        matchesNeeded,
                                                                        keyFrameDatabase,
                                                                        frame,
                                                                        relativeFramePoses);

            if (mapInitializedSuccessfully)
            {
                if (!frame.mTcw.empty() && frame.mpReferenceKF)
                {
                    cv::Mat Tcr = frame.mTcw * frame.mpReferenceKF->GetPoseInverse();
                    relativeFramePoses.push_back(Tcr);
                }
                else if (relativeFramePoses.size())
                {
                    relativeFramePoses.push_back(relativeFramePoses.back());
                }
                return true;
            }
        }
    }
    return false;
}

bool LuluSLAM::createInitialMapMonocular(initializerData &iniData,
                                         SLAMLatestState &last,
                                         ORBVocabulary *voc,
                                         WAIMap *map,
                                         LocalMapping *lmapper,
                                         LoopClosing *loopCloser,
                                         localMap &lmap,
                                         int mapPointsNeeded,
                                         WAIKeyFrameDB*    keyFrameDatabase,
                                         WAIFrame &frame,
                                         list<cv::Mat>&    relativeFramePoses)
{
    //ghm1: reset nNextId to 0! This is important otherwise the first keyframe cannot be identified via its id and a lot of stuff gets messed up!
    //One problem we identified is in UpdateConnections: the first one is not allowed to have a parent,
    //because the second will set the first as a parent too. We get problems later during culling.
    //This also fixes a problem in first GlobalBundleAdjustment which messed up the map after a reset.
    WAIKeyFrame::nNextId = 0;

    // Create KeyFrames
    WAIKeyFrame* pKFini = new WAIKeyFrame(iniData.initialFrame, map, keyFrameDatabase);
    WAIKeyFrame* pKFcur = new WAIKeyFrame(frame, map, keyFrameDatabase);

    pKFini->ComputeBoW(voc);
    pKFcur->ComputeBoW(voc);

    // Insert KFs in the map
    map->AddKeyFrame(pKFini);
    map->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < iniData.iniMatches.size(); i++)
    {
        if (iniData.iniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(iniData.iniPoint3D[i]);

        WAIMapPoint* pMP = new WAIMapPoint(worldPos, pKFcur, map);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, iniData.iniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, iniData.iniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        frame.mvpMapPoints[iniData.iniMatches[i]] = pMP;
        frame.mvbOutlier[iniData.iniMatches[i]]   = false;

        //Add to Map
        map->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    //cout << "New Map created with " << _map->MapPointsInMap() << " points" << endl;

    // Bundle Adjustment
    Optimizer::GlobalBundleAdjustemnt(map, 20);

    // Set median depth to 1
    float medianDepth    = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < mapPointsNeeded)
    {
        WAI_LOG("Wrong initialization, reseting...");
        //reset();
        lmapper->RequestReset();
        loopCloser->RequestReset();
        keyFrameDatabase->clear();
        map->clear();
        lmap.keyFrames.clear();
        lmap.mapPoints.clear();

        WAIKeyFrame::nNextId            = 0;
        WAIFrame::nNextId               = 0;
        WAIFrame::mbInitialComputations = true;
        WAIMapPoint::nNextId            = 0;
        last.lastFrame = WAIFrame();
        last.lastFrameId = 0;
        last.lastRelocFrameId = 0;

        relativeFramePoses.clear();

        return false;
    }

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
        }
    }

    lmapper->InsertKeyFrame(pKFini);
    lmapper->InsertKeyFrame(pKFcur);

    frame.SetPose(pKFcur->GetPose());
    last.lastFrameId = frame.mnId;
    last.lastKeyFrame = pKFcur;

    lmap.keyFrames.push_back(pKFcur);
    lmap.keyFrames.push_back(pKFini);
    lmap.mapPoints = map->GetAllMapPoints();

    lmap.refKF          = pKFcur;
    frame.mpReferenceKF = pKFcur;

    last.lastFrame = WAIFrame(frame);

    map->SetReferenceMapPoints(lmap.mapPoints);

    map->mvpKeyFrameOrigins.push_back(pKFini);

    return true;
}



//Assume state is already tracking and map is already updated
bool LuluSLAM::trackingAndMapping(cv::Mat &camera,
                                  cv::Mat &distortion,
                                  WAIMap *map,
                                  WAIKeyFrameDB* keyFrameDatabase,
                                  SLAMLatestState &last,
                                  localMap &localMap,
                                  LocalMapping *localMapper,
                                  WAIFrame &frame,
                                  list<cv::Mat>& relativeFramePoses,
                                  cv::Mat &pose)
{
    std::unique_lock<std::mutex> lock(map->mMutexMapUpdate, std::defer_lock);
    lock.lock();

    int inliner = 0;

    if (track(frame, last, localMap, relativeFramePoses) || relocalization(frame, last, map, keyFrameDatabase))
    {
        updateLocalMap(frame, localMap);
        if (!localMap.keyFrames.empty())
        {
            frame.mpReferenceKF = localMap.refKF;

            inliner = matchLocalMapPoints(localMap, last, frame);
            if (frame.mnId < last.lastRelocFrameId + MAX_FRAMES && inliner < 50 || inliner < 30)
            {
                return false;
            }
        }
    }

    // Update motion model
    if (!last.lastFrame.mTcw.empty())
    {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        last.lastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3)); //mRwc
        const auto& cc = last.lastFrame.GetCameraCenter(); //this is the translation of the frame w.r.t the world
        cc.copyTo(LastTwc.rowRange(0, 3).col(3));
        last.velocity = frame.mTcw * LastTwc;

        pose = frame.mTcw.clone();
        //Keep pose in an other var?

        // Clean MapPoint without matches
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

        if (needNewKeyFrame(map, localMap, localMapper, last, frame, inliner))
        {
            createNewKeyFrame(localMapper, localMap, map, keyFrameDatabase, last, frame);
        }

        for (int i = 0; i < frame.N; i++)
        {
            if (frame.mvpMapPoints[i] && frame.mvbOutlier[i])
            {
                frame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
            }
        }
    }
    else
    {
        last.velocity = cv::Mat();
    }

    last.lastFrame = WAIFrame(frame);
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (frame.mpReferenceKF && !frame.mTcw.empty())
    {
        cv::Mat Tcr = frame.mTcw * frame.mpReferenceKF->GetPoseInverse(); //Tcr = Tcw * Twr (current wrt reference = world wrt current * reference wrt world
                                                                          //relative frame poses are used to refer a frame to reference frame

        relativeFramePoses.push_back(Tcr);
    }
    else if (relativeFramePoses.size())
    {
        // This can happen if tracking is lost
        relativeFramePoses.push_back(relativeFramePoses.back());
    }
}

void LuluSLAM::createNewKeyFrame(LocalMapping* localMapper,
                                 localMap& lmap,
                                 WAIMap* map,
                                 WAIKeyFrameDB* keyFrameDatabase,
                                 SLAMLatestState& last,
                                 WAIFrame& frame)
{
    if (!localMapper->SetNotStop(true))
        return;

    WAIKeyFrame* pKF = new WAIKeyFrame(frame, map, keyFrameDatabase);

    lmap.refKF = pKF;
    frame.mpReferenceKF = pKF;
    localMapper->InsertKeyFrame(pKF);
    localMapper->SetNotStop(false);

    last.lastKeyFrame = pKF;
}

bool LuluSLAM::needNewKeyFrame(WAIMap *map, localMap &lmap, LocalMapping* lmapper, SLAMLatestState &last, WAIFrame &frame, int nInliners)
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (lmapper->isStopped() || lmapper->stopRequested())
        return false;

    const int nKFs = map->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (frame.mnId < last.lastKeyFrame->mnId + MAX_FRAMES && nKFs > MAX_FRAMES)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    int nRefMatches = lmap.refKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = lmapper->AcceptKeyFrames();

    // Thresholds
    float thRefRatio = 0.9f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = frame.mnId >= last.lastKeyFrame->mnId + MAX_FRAMES;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (frame.mnId >= last.lastKeyFrame->mnId + MIN_FRAMES && bLocalMappingIdle);
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
            lmapper->InterruptBA();
            return false;
        }
    }
    else
    {
        return false;
    }
}

//Assume state is already tracking and map is already updated
bool LuluSLAM::track(WAIFrame &frame,
                     SLAMLatestState &slam,
                     localMap &localMap,
                     list<cv::Mat>& relativeFramePoses)
{
    if (!trackWithMotionModel(slam, frame, relativeFramePoses))
    {
        return trackReferenceKeyFrame(slam, localMap, frame);
    }
    return true;
}

bool LuluSLAM::relocalization(WAIFrame& currentFrame, SLAMLatestState &slam, WAIMap* waiMap, WAIKeyFrameDB* keyFrameDatabase)
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
                    bMatch = true;
                    break;
                }
            }
        }
    }

    AVERAGE_TIMING_STOP("relocalization");
    return bMatch;
}


bool LuluSLAM::trackReferenceKeyFrame(SLAMLatestState &last, localMap &map, WAIFrame &frame)
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
    frame.SetPose(last.lastFrame.mTcw);

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

int LuluSLAM::matchLocalMapPoints(localMap &lmap, SLAMLatestState &last, WAIFrame &frame)
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
    for (vector<WAIMapPoint*>::iterator vit = lmap.mapPoints.begin(), vend = lmap.mapPoints.end(); vit != vend; vit++)
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
        if (frame.mnId -1 == last.lastRelocFrameId)
            th = 5;
        matcher.SearchByProjection(frame, lmap.mapPoints, th);
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
            }
        }
    }

    AVERAGE_TIMING_STOP("trackLocalMap");
    return matchesInliers;
}

void LuluSLAM::updateLocalMap(WAIFrame&     frame,
                              localMap&     lmap)
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

    int          max    = 0;
    lmap.refKF = static_cast<WAIKeyFrame*>(NULL);

    lmap.keyFrames.clear();
    lmap.keyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (auto it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        WAIKeyFrame* pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)   //Should be computed with parent and child added to localmap
        {
            max    = it->second;
            lmap.refKF = pKF;
        }

        lmap.keyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = frame.mnId; // <==== UHHH WHAT IS THAT???
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (auto itKF = lmap.keyFrames.begin(); itKF != lmap.keyFrames.end(); itKF++)
    {
        // Limit the number of keyframes
        if (lmap.keyFrames.size() > 80)
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
                    lmap.keyFrames.push_back(pNeighKF);
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
                    lmap.keyFrames.push_back(pChildKF);
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
                lmap.keyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = frame.mnId;
                break;
            }
        }
    }

    lmap.mapPoints.clear();
    for (vector<WAIKeyFrame*>::const_iterator itKF = lmap.keyFrames.begin(), itEndKF = lmap.keyFrames.end(); itKF != itEndKF; itKF++)
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
                lmap.mapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = frame.mnId;
            }
        }
    }
}

bool LuluSLAM::trackWithMotionModel(SLAMLatestState& last, WAIFrame& frame, list<cv::Mat>& relativeFramePoses)
{
    AVERAGE_TIMING_START("trackWithMotionModel");

    if (last.velocity.empty() || frame.mnId == last.lastRelocFrameId + 1)
        return false;

    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    WAIKeyFrame * pRef = last.lastFrame.mpReferenceKF;
    last.lastFrame.SetPose(relativeFramePoses.back() * pRef->GetPose());


    //this adds the motion differnce between the last and the before-last frame to the pose of the last frame to estimate the position of the current frame
    frame.SetPose(last.velocity * last.lastFrame.mTcw);

    fill(frame.mvpMapPoints.begin(), frame.mvpMapPoints.end(), static_cast<WAIMapPoint*>(NULL));

    // Project points seen in previous frame
    int th       = 15;
    int nmatches = matcher.SearchByProjection(frame, last.lastFrame, th, true);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(frame.mvpMapPoints.begin(), frame.mvpMapPoints.end(), static_cast<WAIMapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(frame, last.lastFrame, 2 * th, true);
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

void LuluSLAM::requestStateIdle()
{
    std::unique_lock<std::mutex> guard(mMutexStates);
    mLocalMapping->RequestStop();
    while (!mLocalMapping->isStopped())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    mState = TrackingState_Idle;
}

bool LuluSLAM::hasStateIdle()
{
    return (mState == TrackingState_Idle);
}

WAIMap* LuluSLAM::getMap()
{
    return mGlobalMap;
}

WAIKeyFrameDB* LuluSLAM::getKfDB()
{
    return mKeyFrameDatabase;
}

bool LuluSLAM::retainImage()
{
    return false;
}

void LuluSLAM::resume()
{
    mLocalMapping->Release();
    mState = TrackingState_TrackingLost;
}

void LuluSLAM::setInitialized(bool b)
{
    mInitialized = true;
    mState = TrackingState_TrackingLost;
}

bool LuluSLAM::isInitialized()
{
    return mInitialized;
}


void LuluSLAM::reset()
{
    WAI_LOG("System Reseting");

    mLocalMapping->RequestReset();

    mLoopClosing->RequestReset();


    // Clear BoW Database
    mKeyFrameDatabase->clear();

    // Clear Map (this erase MapPoints and KeyFrames)
    mGlobalMap->clear();

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;

    if (mIniData.initializer)
    {
        delete mIniData.initializer;
        mIniData.initializer = static_cast<Initializer*>(nullptr);
    }

    mRelativeFramePoses.clear();

    mLast.lastKeyFrame = nullptr;
    mLast.lastRelocFrameId = 0;
    mLast.lastFrameId = 0;

    mLmap.refKF = nullptr;
    mLmap.mapPoints.clear();
    mLmap.keyFrames.clear();

    mState = TrackingState_Initializing;
}

KPextractor* LuluSLAM::getKPextractor()
{
    return mExtractor;
}

WAIFrame* LuluSLAM::getLastFrame()
{
    return &mLastFrame;
}

void LuluSLAM::drawKeyPointInfo(WAIFrame &frame, cv::Mat& image)
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


void LuluSLAM::drawKeyPointMatches(WAIFrame &frame, cv::Mat& image)
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

std::vector<WAIMapPoint*> LuluSLAM::getMapPoints()
{
    return mGlobalMap->GetAllMapPoints();
}

std::vector<WAIKeyFrame*> LuluSLAM::getKeyFrames()
{
    return mGlobalMap->GetAllKeyFrames();
}

std::vector<WAIMapPoint*> LuluSLAM::getLocalMapPoints()
{
    return mLmap.mapPoints;
}

std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> LuluSLAM::getMatchedCorrespondances(WAIFrame& frame)
{
    std::vector<cv::Vec3f> points3d;
    std::vector<cv::Vec2f> points2d;

    for (int i = 0; i < frame.N; i++)
    {
        WAIMapPoint* mp = frame.mvpMapPoints[i];
        if (mp)
        {
            if (!frame.mvbOutlier[i])
            {
                if (mp->Observations() > 0)
                {
                    WAI::V3   _v = mp->worldPosVec();
                    cv::Vec3f v;
                    v[0] = _v.x;
                    v[1] = _v.y;
                    v[2] = _v.z;
                    points3d.push_back(v);
                    points2d.push_back(frame.mvKeysUn[i].pt);
                }
            }
        }
    }

    return std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>>(points3d, points2d);
}

std::vector<WAIMapPoint*> LuluSLAM::getMatchedMapPoints(WAIFrame* frame)
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

        //numbers
        //add tracking state
std::string LuluSLAM::getPrintableState()
{
    switch (mState)
    {
        case TrackingState_Idle:
            return std::string("TrackingState_Idle\n");
            break;
        case TrackingState_Initializing:
            return std::string("TrackingState_Initializing");
            break;
        case TrackingState_None:
            return std::string("TrackingState_None");
            break;
        case TrackingState_TrackingLost:
            return std::string("TrackingState_TrackingLost");
            break;
        case TrackingState_TrackingOK:
            return std::string("TrackingState_TrackingOK");
            break;
    }
}

int LuluSLAM::getKeyPointCount()
{
    return mLastFrame.N;
}

int LuluSLAM::getKeyFrameCount()
{
    return mGlobalMap->KeyFramesInMap();
}

int LuluSLAM::getMapPointCount()
{
    return mGlobalMap->MapPointsInMap();
}

cv::Mat LuluSLAM::getPose()
{
    return mCameraExtrinsic;
}
