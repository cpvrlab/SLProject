#include <WAIModeOrbSlam2.h>

WAI::ModeOrbSlam2::ModeOrbSlam2(SensorCamera* camera,
                                bool          serial,
                                bool          retainImg,
                                bool          onlyTracking,
                                bool          trackOptFlow,
                                std::string   orbVocFile)
  : Mode(WAI::ModeType_ORB_SLAM2),
    _serial(serial),
    _retainImg(retainImg),
    _onlyTracking(onlyTracking),
    _trackOptFlow(trackOptFlow),
    _camera(camera)
{

    //load visual vocabulary for relocalization
    WAIOrbVocabulary::initialize(orbVocFile);
    mpVocabulary = WAIOrbVocabulary::get();

    //instantiate and load slam map
    mpKeyFrameDatabase = new WAIKeyFrameDB(*mpVocabulary);

    _map = new WAIMap("Map");

    //setup file system and check for existing files
    //WAIMapStorage::init();
    //make new map
    //WAIMapStorage::newMap();

    if (_map->KeyFramesInMap())
        _initialized = true;
    else
        _initialized = false;

    int   nFeatures    = 1000;
    float fScaleFactor = 1.2;
    int   nLevels      = 8;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;

    //instantiate Orb extractor
    //_extractor        = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    //pIniORBextractor = new ORB_SLAM2::ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    _extractor        = new ORB_SLAM2::SURFextractor(1500);
    mpIniORBextractor = new ORB_SLAM2::SURFextractor(1000);
    //instantiate local mapping
    mpLocalMapper = new ORB_SLAM2::LocalMapping(_map, 1, mpVocabulary);
    mpLoopCloser  = new ORB_SLAM2::LoopClosing(_map, mpKeyFrameDatabase, mpVocabulary, false, false);

    mpLocalMapper->SetLoopCloser(mpLoopCloser);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    if (!_serial)
    {
        mptLocalMapping = new std::thread(&LocalMapping::Run, mpLocalMapper);
        mptLoopClosing  = new std::thread(&LoopClosing::Run, mpLoopCloser);
    }

    _camera->subscribeToUpdate(this);
    _state = TrackingState_Initializing;

    _pose = cv::Mat(4, 4, CV_32F);
}

WAI::ModeOrbSlam2::~ModeOrbSlam2()
{
    if (!_serial)
    {
        mpLocalMapper->RequestFinish();
        mpLoopCloser->RequestFinish();

        // Wait until all thread have effectively stopped
        mptLocalMapping->join();
        mptLoopClosing->join();
    }

    if (_extractor)
    {
        delete _extractor;
    }
    if (mpIniORBextractor)
    {
        delete mpIniORBextractor;
    }
    if (mpLocalMapper)
    {
        delete mpLocalMapper;
    }
    if (mpLoopCloser)
    {
        delete mpLoopCloser;
    }
}

bool WAI::ModeOrbSlam2::getPose(cv::Mat* pose)
{
    bool result = 0;

    if (_state == TrackingState_TrackingOK && _poseSet)
    {
        *pose  = _pose;
        result = 1;
    }

    return result;
}

void WAI::ModeOrbSlam2::notifyUpdate()
{
    //__android_log_print(ANDROID_LOG_INFO, "lib-WAI", "update notified");
    stateTransition();

    switch (_state)
    {
        case TrackingState_Initializing:
        {
            initialize();
        }
        break;

        case TrackingState_TrackingOK:
        case TrackingState_TrackingLost:
        {
            //relocalize or track 3d points
            track3DPts();
        }
        break;

        case TrackingState_Idle:
        case TrackingState_None:
        default:
        {
        }
        break;
    }
}

uint32_t WAI::ModeOrbSlam2::getMapPointCount()
{
    uint32_t result = _map->MapPointsInMap();

    return result;
}

uint32_t WAI::ModeOrbSlam2::getMapPointMatchesCount()
{
    uint32_t result = mnMatchesInliers;

    return result;
}

uint32_t WAI::ModeOrbSlam2::getKeyFrameCount()
{
    uint32_t result = _map->KeyFramesInMap();

    return result;
}

std::string WAI::ModeOrbSlam2::getLoopCloseStatus()
{
    std::string result = mpLoopCloser->getStatusString();

    return result;
}

uint32_t WAI::ModeOrbSlam2::getLoopCloseCount()
{
    uint32_t result = _map->getNumLoopClosings();

    return result;
}

uint32_t WAI::ModeOrbSlam2::getKeyFramesInLoopCloseQueueCount()
{
    uint32_t result = mpLoopCloser->numOfKfsInQueue();

    return result;
}

int WAI::ModeOrbSlam2::getNMapMatches()
{
    std::lock_guard<std::mutex> guard(_nMapMatchesLock);
    return mnMatchesInliers;
}
//-----------------------------------------------------------------------------
int WAI::ModeOrbSlam2::getNumKeyFrames()
{
    std::lock_guard<std::mutex> guard(_mapLock);
    return _map->KeyFramesInMap();
}
//-----------------------------------------------------------------------------
float WAI::ModeOrbSlam2::poseDifference()
{
    std::lock_guard<std::mutex> guard(_poseDiffLock);
    return _poseDifference;
}

float WAI::ModeOrbSlam2::getMeanReprojectionError()
{
    return _meanReprojectionError;
}

std::string WAI::ModeOrbSlam2::getPrintableState()
{
    std::string printableState = "";

    switch (_state)
    {
        case TrackingState_Initializing:
        {
            printableState = "INITIALIZING";
        }
        break;

        case TrackingState_Idle:
        {
            printableState = "IDLE";
        }
        break;

        case TrackingState_TrackingLost:
        {
            printableState = "TRACKING_LOST"; //motion model tracking
        }
        break;

        case TrackingState_TrackingOK:
        {
            printableState = "TRACKING_OK";
        }
        break;

        case TrackingState_None:
        {
            printableState = "TRACKING_NONE";
        }
        break;

        default:
        {
            printableState = "";
        }
        break;
    }

    return printableState;
}

std::string WAI::ModeOrbSlam2::getPrintableType()
{
    switch (_trackingType)
    {
        case TrackingType_MotionModel:
            return "Motion Model";
        case TrackingType_OptFlow:
            return "Optical Flow";
        case TrackingType_ORBSLAM:
            return "ORB-SLAM";
        case TrackingType_None:
        default:
            return "None";
    }
}

std::vector<WAIMapPoint*> WAI::ModeOrbSlam2::getMapPoints()
{
    std::lock_guard<std::mutex> guard(_mapLock);

    std::vector<WAIMapPoint*> result = _map->GetAllMapPoints();

    return result;
}

std::vector<WAIMapPoint*> WAI::ModeOrbSlam2::getMatchedMapPoints()
{
    std::lock_guard<std::mutex> guard(_mapLock);

    std::vector<WAIMapPoint*> result;

    if (_optFlowOK)
    {
        result = _optFlowMapPtsLastFrame;
    }
    else
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        result.push_back(mCurrentFrame.mvpMapPoints[i]);
                }
            }
        }
    }

    return result;
}

std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> WAI::ModeOrbSlam2::getMatchedCorrespondances()
{
    std::lock_guard<std::mutex> guard(_mapLock);

    std::vector<cv::Vec3f> points3d;
    std::vector<cv::Vec2f> points2d;

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        WAIMapPoint* mp = mCurrentFrame.mvpMapPoints[i];
        if (mp)
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                if (mp->Observations() > 0)
                {
                    WAI::V3   _v = mp->worldPosVec();
                    cv::Vec3f v;
                    v[0] = _v.x;
                    v[1] = _v.y;
                    v[2] = _v.z;
                    points3d.push_back(v);
                    points2d.push_back(mCurrentFrame.mvKeysUn[i].pt);
                }
            }
        }
    }

    return std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>>(points3d, points2d);
}

std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>> WAI::ModeOrbSlam2::getCorrespondances()
{
    std::lock_guard<std::mutex> guard(_mapLock);

    std::vector<cv::Vec3f> points3d;
    std::vector<cv::Vec2f> points2d;

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        WAIMapPoint* mp = mCurrentFrame.mvpMapPoints[i];
        if (mp)
        {
            WAI::V3   _v = mp->worldPosVec();
            cv::Vec3f v;
            v[0] = _v.x;
            v[1] = _v.y;
            v[2] = _v.z;
            points3d.push_back(v);
            points2d.push_back(mCurrentFrame.mvKeys[i].pt);
        }
    }

    return std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec2f>>(points3d, points2d);
}

std::vector<WAIMapPoint*> WAI::ModeOrbSlam2::getLocalMapPoints()
{
    std::lock_guard<std::mutex> guard(_mapLock);

    std::vector<WAIMapPoint*> result = mvpLocalMapPoints;

    return result;
}

std::vector<WAIKeyFrame*> WAI::ModeOrbSlam2::getKeyFrames()
{
    std::lock_guard<std::mutex> guard(_mapLock);

    std::vector<WAIKeyFrame*> result = _map->GetAllKeyFrames();

    return result;
}

bool WAI::ModeOrbSlam2::getTrackOptFlow()
{
    std::lock_guard<std::mutex> guard(_optFlowLock);
    return _trackOptFlow;
}

void WAI::ModeOrbSlam2::setTrackOptFlow(bool flag)
{
    std::lock_guard<std::mutex> guard(_optFlowLock);
    _trackOptFlow = flag;
    _optFlowOK    = false;
}

void WAI::ModeOrbSlam2::disableMapping()
{
    _onlyTracking = true;
    if (!_serial)
    {
        mpLocalMapper->RequestStop();
        while (!mpLocalMapper->isStopped())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    mpLocalMapper->InterruptBA();
}

void WAI::ModeOrbSlam2::enableMapping()
{
    _onlyTracking = false;
    resume();
}

void WAI::ModeOrbSlam2::stateTransition()
{
    std::lock_guard<std::mutex> guard(_mutexStates);

    //store last state
    //mLastProcessedState = _state;

    if (_idleRequested)
    {
        _state         = TrackingState_Idle;
        _idleRequested = false;
    }
    else if (_state == TrackingState_Idle)
    {
        if (_resumeRequested)
        {
            if (!_initialized)
            {
                _state = TrackingState_Initializing;
            }
            else
            {
                _state = TrackingState_TrackingLost;
            }

            _resumeRequested = false;
        }
    }
    else if (_state == TrackingState_Initializing)
    {
        if (_initialized)
        {
            if (_bOK)
            {
                _state = TrackingState_TrackingOK;
            }
            else
            {
                _state = TrackingState_TrackingLost;
            }
        }
    }
    else if (_state == TrackingState_TrackingOK)
    {
        if (!_bOK)
        {
            _state = TrackingState_TrackingLost;
        }
    }
    else if (_state == TrackingState_TrackingLost)
    {
        if (_bOK)
        {
            _state = TrackingState_TrackingOK;
        }
    }
}

void WAI::ModeOrbSlam2::initialize()
{
    //1. if there are more than 100 keypoints in the current frame, the Initializer is instantiated
    //2. if there are less than 100 keypoints in the next frame, the Initializer is deinstantiated again
    //3. else if there are more than 100 keypoints we try to match the keypoints in the current with the initial frame
    //4. if we found less than 100 matches between the current and the initial keypoints, the Initializer is deinstantiated
    //5. else we try to initializer: that means a homograhy and a fundamental matrix are calculated in parallel and 3D points are triangulated initially
    //6. if the initialization (by homograhy or fundamental matrix) was successful an inital map is created:
    //  - two keyframes are generated from the initial and the current frame and added to keyframe database and map
    //  - a mappoint is instantiated from the triangulated 3D points and all necessary members are calculated (distinctive descriptor, depth and normal, add observation reference of keyframes)
    //  - a global bundle adjustment is applied
    //  - the two new keyframes are added to the local mapper and the local mapper is started twice
    //  - the tracking state is changed to TRACKING/INITIALIZED

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(_map->mMutexMapUpdate, std::defer_lock);
    if (!_serial)
    {
        lock.lock();
    }

    cv::Mat cameraMat     = _camera->getCameraMatrix();
    cv::Mat distortionMat = _camera->getDistortionMatrix();
    mCurrentFrame         = WAIFrame(_camera->getImageGray(),
                             0.0,
                             mpIniORBextractor,
                             cameraMat,
                             distortionMat,
                             mpVocabulary,
                             _retainImg);

    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            mInitialFrame = WAIFrame(mCurrentFrame);
            mLastFrame    = WAIFrame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            //ghm1: we store the undistorted keypoints of the initial frame in an extra vector
            //todo: why not using mInitialFrame.mvKeysUn????
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            // TODO(jan): is this necessary?
            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new ORB_SLAM2::Initializer(mCurrentFrame, 1.0, 200);
            //ghm1: clear mvIniMatches. it contains the index of the matched keypoint in the current frame
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int        nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        for (unsigned int i = 0; i < mInitialFrame.mvKeys.size(); i++)
        {
            cv::rectangle(_camera->getImageRGB(),
                          mInitialFrame.mvKeys[i].pt,
                          cv::Point(mInitialFrame.mvKeys[i].pt.x + 3, mInitialFrame.mvKeys[i].pt.y + 3),
                          cv::Scalar(0, 0, 255));
        }

        //ghm1: decorate image with tracked matches
        for (unsigned int i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] >= 0)
            {
                cv::line(_camera->getImageRGB(),
                         mInitialFrame.mvKeys[i].pt,
                         mCurrentFrame.mvKeys[mvIniMatches[i]].pt,
                         cv::Scalar(0, 255, 0));
            }
        }

        cv::Mat      Rcw;            // Current Camera Rotation
        cv::Mat      tcw;            // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            bool mapInitializedSuccessfully = createInitialMapMonocular();
            if (mapInitializedSuccessfully)
            {
                //mark tracking as initialized
                _initialized = true;
                _bOK         = true;
                //_state       = TrackingState_TrackingOK;
            }

            //ghm1: in the original implementation the initialization is defined in the track() function and this part is always called at the end!
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if (!mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF)
            {
                cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr);
                mlpReferences.push_back(mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(_state == TrackingState_TrackingLost);
            }
            else if (mlRelativeFramePoses.size())
            {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(_state == TrackingState_TrackingLost);
            }
        }
    }
}

bool WAI::ModeOrbSlam2::posInGrid(const cv::KeyPoint& kp, int& posX, int& posY, int minX, int minY)
{
    posX = (int)round((kp.pt.x - minX) * _optFlowGridElementWidthInv);
    posY = (int)round((kp.pt.y - minY) * _optFlowGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= OPTFLOW_GRID_COLS || posY < 0 || posY >= OPTFLOW_GRID_ROWS)
        return false;

    return true;
}

void WAI::ModeOrbSlam2::track3DPts()
{
    cv::Mat cameraMat     = _camera->getCameraMatrix();
    cv::Mat distortionMat = _camera->getDistortionMatrix();
    mCurrentFrame         = WAIFrame(_camera->getImageGray(),
                             0.0,
                             _extractor,
                             cameraMat,
                             distortionMat,
                             mpVocabulary,
                             _retainImg);

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(_map->mMutexMapUpdate, std::defer_lock);
    if (!_serial)
    {
        lock.lock();
    }

    //mLastProcessedState = mState;
    //bool bOK;
    _bOK = false;
    //trackingType = TrackingType_None;

    if (!_onlyTracking)
    {
        // Local Mapping is activated. This is the normal behaviour, unless
        // you explicitly activate the "only tracking" mode.

        if (_state == TrackingState_TrackingOK)
        {
            // Local Mapping might have changed some MapPoints tracked in last frame
            checkReplacedInLastFrame();

            if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                _bOK = trackReferenceKeyFrame();
                //trackingType = TrackingType_ORBSLAM;
            }
            else
            {
                _bOK = trackWithMotionModel();
                //trackingType = TrackingType_MotionModel;

                if (!_bOK)
                {
                    _bOK = trackReferenceKeyFrame();
                    //trackingType = TrackingType_ORBSLAM;
                }
            }
        }
        else
        {
            _bOK = relocalization();
        }
    }
    else
    {
        // Localization Mode: Local Mapping is deactivated

        if (_state == TrackingState_TrackingLost)
        {
            _bOK       = relocalization();
            _optFlowOK = false;
            //cout << "Relocalization: " << bOK << endl;
        }
        else
        {
            //if NOT visual odometry tracking
            if (!mbVO) // In last frame we tracked enough MapPoints from the Map
            {
                if (!mVelocity.empty())
                { //we have a valid motion model
                    _bOK = trackWithMotionModel();
                    //trackingType = TrackingType_MotionModel;
                    //cout << "TrackWithMotionModel: " << bOK << endl;
                }
                else
                {
                    //we have NO valid motion model
                    // All keyframes that observe a map point are included in the local map.
                    // Every current frame gets a reference keyframe assigned which is the keyframe
                    // from the local map that shares most matches with the current frames local map points matches.
                    // It is updated in UpdateLocalKeyFrames().
                    _bOK = trackReferenceKeyFrame();
                    //trackingType = TrackingType_ORBSLAM;
                    //cout << "TrackReferenceKeyFrame" << endl;
                }
            }
            else // In last frame we tracked mainly "visual odometry" points.
            {
                // We compute two camera poses, one from motion model and one doing relocalization.
                // If relocalization is sucessfull we choose that solution, otherwise we retain
                // the "visual odometry" solution.
                bool                 bOKMM    = false;
                bool                 bOKReloc = false;
                vector<WAIMapPoint*> vpMPsMM;
                vector<bool>         vbOutMM;
                cv::Mat              TcwMM;
                if (!mVelocity.empty())
                {
                    bOKMM   = trackWithMotionModel();
                    vpMPsMM = mCurrentFrame.mvpMapPoints;
                    vbOutMM = mCurrentFrame.mvbOutlier;
                    TcwMM   = mCurrentFrame.mTcw.clone();
                }
                bOKReloc = relocalization();

                //relocalization method is not valid but the velocity model method
                if (bOKMM && !bOKReloc)
                {
                    mCurrentFrame.SetPose(TcwMM);
                    mCurrentFrame.mvpMapPoints = vpMPsMM;
                    mCurrentFrame.mvbOutlier   = vbOutMM;

                    if (mbVO)
                    {
                        for (int i = 0; i < mCurrentFrame.N; i++)
                        {
                            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                            {
                                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                            }
                        }
                    }
                }
                else if (bOKReloc)
                {
                    mbVO = false;
                }

                _bOK = bOKReloc || bOKMM;
                //trackingType = TrackingType_None;
            }
        }
    }

    // If we have an initial estimation of the camera pose and matching. Track the local map.
    if (!_onlyTracking)
    {
        if (_bOK)
        {
            _bOK = trackLocalMap();
        }
    }
    else
    {
        // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
        // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
        // the camera we will use the local map again.
        if (_bOK && !mbVO)
            _bOK = trackLocalMap();
    }

    if (_trackOptFlow && _bOK && _state == TrackingState_TrackingOK)
    {
        //We always run the optical flow additionally, because it gives
        //a more stable pose. We use this pose if successful.
        _optFlowOK = trackWithOptFlow();
    }

    //if (bOK)
    //    mState = OK;
    //else
    //    mState = LOST;

    // If tracking were good
    if (_bOK)
    {
        // Update motion model
        if (!mLastFrame.mTcw.empty())
        {
            //cout << "mLastFrame.mTcw: " << mLastFrame.mTcw << endl;
            cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
            //cout << "LastTwc eye: " << LastTwc << endl;
            mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3)); //mRwc
            //cout << "LastTwc rot: " << LastTwc << endl;
            const auto& cc = mLastFrame.GetCameraCenter(); //this is the translation of the frame w.r.t the world
            //cout << cc << endl;
            cc.copyTo(LastTwc.rowRange(0, 3).col(3));
            //cout << "LastTwc total: " << LastTwc << endl;
            //this concatenates the motion difference between the last and the before-last frame (so it is no real velocity but a transformation)
            mVelocity = mCurrentFrame.mTcw * LastTwc;
        }
        else
        {
            mVelocity = cv::Mat();
        }

        //set current pose
        {
            cv::Mat Tcw;
            if (_optFlowOK)
            {
                Tcw = _optFlowTcw.clone();
            }
            else
            {
                Tcw = mCurrentFrame.mTcw.clone();
            }

#if 0
            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            //inversion
            //auto Tcw = mCurrentFrame.mTcw.clone();
            Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            for (int col = 0; col < 3; col++)
            {
                for (int row = 0; row < 3; row++)
                {
                    _pose.at<float>(row, col) = Rwc.at<float>(row, col);
                }
            }

            for (int row = 0; row < 3; row++)
            {
                _pose.at<float>(row, 3) = twc.at<float>(row, 0);
            }

            _pose.at<float>(3, 3) = 1.0f;
#else
            _pose = Tcw;
#endif
            _poseSet = true;

            //conversion to SLMat4f
            //SLMat4f slMat((SLfloat)Rwc.at<float>(0, 0), (SLfloat)Rwc.at<float>(0, 1), (SLfloat)Rwc.at<float>(0, 2), (SLfloat)twc.at<float>(0, 0), (SLfloat)Rwc.at<float>(1, 0), (SLfloat)Rwc.at<float>(1, 1), (SLfloat)Rwc.at<float>(1, 2), (SLfloat)twc.at<float>(1, 0), (SLfloat)Rwc.at<float>(2, 0), (SLfloat)Rwc.at<float>(2, 1), (SLfloat)Rwc.at<float>(2, 2), (SLfloat)twc.at<float>(2, 0), 0.0f, 0.0f, 0.0f, 1.0f);
            //slMat.rotate(180, 1, 0, 0);
            // set the object matrix of this object (its a SLCamera)
            //_node->om(slMat);
        }

        // Clean VO matches
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            WAIMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (pMP->Observations() < 1)
                {
                    mCurrentFrame.mvbOutlier[i]   = false;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
                }
            }
        }

        //ghm1: manual local mapping of current frame
        if (needNewKeyFrame())
        {
            createNewKeyFrame();

            if (_serial)
            {
                //call local mapper
                mpLocalMapper->RunOnce();
                //normally the loop closing would feed the keyframe database, but we do it here
                //mpKeyFrameDatabase->add(mpLastKeyFrame);

                //loop closing
                mpLoopCloser->RunOnce();
            }

            //update visualization of map, it may have changed because of global bundle adjustment.
            //map points will be updated with next decoration.
            _mapHasChanged = true;
        }

        // We allow points with high innovation (considererd outliers by the Huber Function)
        // pass to the new keyframe, so that bundle adjustment will finally decide
        // if they are outliers or not. We don't want next frame to estimate its position
        // with those points so we discard them in the frame.
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
            }
        }
    }

    if (!mCurrentFrame.mpReferenceKF)
    {
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    decorate();

    mLastFrame = WAIFrame(mCurrentFrame);

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (mCurrentFrame.mpReferenceKF && !mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse(); //Tcr = Tcw * Twr (current wrt reference = world wrt current * reference wrt world
                                                                                          //relative frame poses are used to refer a frame to reference frame
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(_state == TrackingState_TrackingLost);
    }
    else if (mlRelativeFramePoses.size() && mlpReferences.size() && mlFrameTimes.size())
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(_state == TrackingState_TrackingLost);
    }
}

bool WAI::ModeOrbSlam2::createInitialMapMonocular()
{
    // Create KeyFrames
    WAIKeyFrame* pKFini = new WAIKeyFrame(mInitialFrame, _map, mpKeyFrameDatabase);
    WAIKeyFrame* pKFcur = new WAIKeyFrame(mCurrentFrame, _map, mpKeyFrameDatabase);

    cout << "pKFini num keypoints: " << mInitialFrame.N << endl;
    cout << "pKFcur num keypoints: " << mCurrentFrame.N << endl;

    pKFini->ComputeBoW(mpVocabulary);
    pKFcur->ComputeBoW(mpVocabulary);

    // Insert KFs in the map
    _map->AddKeyFrame(pKFini);
    _map->AddKeyFrame(pKFcur);
    //mpKeyFrameDatabase->add(pKFini);
    //mpKeyFrameDatabase->add(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        WAIMapPoint* pMP = new WAIMapPoint(worldPos, pKFcur, _map);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]]   = false;

        //Add to Map
        _map->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << _map->MapPointsInMap() << " points" << endl;
    Optimizer::GlobalBundleAdjustemnt(_map, 20);

    // Set median depth to 1
    float medianDepth    = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        reset();
        return false;
    }

    // Scale initial baseline
    cv::Mat Tc2w               = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    //pKFcur->SetPose(Tc2w);
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

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = _map->GetAllMapPoints();

    mpReferenceKF               = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = WAIFrame(mCurrentFrame);

    _map->SetReferenceMapPoints(mvpLocalMapPoints);

    _map->mvpKeyFrameOrigins.push_back(pKFini);

    //mState = OK;
    /*_bOK = true;*/
    //_currentState = TRACK_3DPTS;

    std::cout << pKFini->GetPose() << std::endl;
    std::cout << pKFcur->GetPose() << std::endl;

    //ghm1: run local mapping once
    if (_serial)
    {
        mpLocalMapper->RunOnce();
        mpLocalMapper->RunOnce();
    }

    // Bundle Adjustment
    cout << "Number of Map points after local mapping: " << _map->MapPointsInMap() << endl;

    //ghm1: add keyframe to scene graph. this position is wrong after bundle adjustment!
    //set map dirty, the map will be updated in next decoration
    _mapHasChanged = true;
    return true;
}

void WAI::ModeOrbSlam2::checkReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        WAIMapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            WAIMapPoint* pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool WAI::ModeOrbSlam2::needNewKeyFrame()
{
    if (_onlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = _map->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Thresholds
    float thRefRatio = 0.9f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio) && mnMatchesInliers > 15);

    if ((c1a || c1b) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            std::cout << "[WAITrackedMapping] NeedNewKeyFrame: YES bLocalMappingIdle!" << std::endl;
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            std::cout << "[WAITrackedMapping] NeedNewKeyFrame: NO InterruptBA!" << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "[WAITrackedMapping] NeedNewKeyFrame: NO!" << std::endl;
        return false;
    }
}

void WAI::ModeOrbSlam2::createNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    WAIKeyFrame* pKF = new WAIKeyFrame(mCurrentFrame, _map, mpKeyFrameDatabase);

    mpReferenceKF               = pKF;
    mCurrentFrame.mpReferenceKF = pKF;
    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKF;
}

void WAI::ModeOrbSlam2::reset()
{
    WAI_LOG("System Reseting");

    // Reset Local Mapping
    if (!_serial)
    {
        mpLocalMapper->RequestReset();
    }
    else
    {
        mpLocalMapper->reset();
    }

    //// Reset Loop Closing
    if (!_serial)
    {
        mpLoopCloser->RequestReset();
    }
    else
    {
        mpLoopCloser->reset();
    }

    // Clear BoW Database
    mpKeyFrameDatabase->clear();

    // Clear Map (this erase MapPoints and KeyFrames)
    _map->clear();

    WAIKeyFrame::nNextId = 0;
    WAIFrame::nNextId    = 0;
    _bOK                 = false;
    _initialized         = false;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(nullptr);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    mpLastKeyFrame = nullptr;
    mpReferenceKF  = nullptr;
    mvpLocalMapPoints.clear();
    mvpLocalKeyFrames.clear();
    mnMatchesInliers   = 0;
    mnLastKeyFrameId   = 0;
    mnLastRelocFrameId = 0;

    _state = TrackingState_Initializing;
}

bool WAI::ModeOrbSlam2::isInitialized()
{
    bool result = _initialized;

    return result;
}

void WAI::ModeOrbSlam2::pause()
{
    if (!_serial)
    {
        mpLocalMapper->RequestStop();
        while (!mpLocalMapper->isStopped())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    requestStateIdle();
    while (!hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void WAI::ModeOrbSlam2::resume()
{
    if (!_serial)
    {
        mpLocalMapper->Release();
        //mptLocalMapping = new thread(&LocalMapping::Run, mpLocalMapper);
        //mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);
    }

    requestResume();
}

void WAI::ModeOrbSlam2::requestStateIdle()
{
    {
        std::unique_lock<std::mutex> guard(_mutexStates);
        resetRequests();
        _idleRequested = true;
    }

    stateTransition();
}

void WAI::ModeOrbSlam2::requestResume()
{
    {
        std::unique_lock<std::mutex> guard(_mutexStates);
        resetRequests();
        _resumeRequested = true;
    }

    stateTransition();
}

bool WAI::ModeOrbSlam2::hasStateIdle()
{
    std::unique_lock<std::mutex> guard(_mutexStates);

    bool result = (_state == TrackingState_Idle);

    return result;
}

void WAI::ModeOrbSlam2::resetRequests()
{
    _idleRequested   = false;
    _resumeRequested = false;
}

void WAI::ModeOrbSlam2::findMatches(std::vector<cv::Point2f>& vP2D, std::vector<cv::Point3f>& vP3Dw)
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query WAIKeyFrame Database for keyframe candidates for relocalisation
    vector<WAIKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return;

    //vector<WAIKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->keyFrames();
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    ORBmatcher matcher(0.75, true);

    vector<vector<WAIMapPoint*>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    for (int i = 0; i < nKFs; i++)
    {
        WAIKeyFrame* pKF      = vpCandidateKFs[i];
        int          nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
        if (nmatches < 15)
            continue;
        int idx = 0;

        for (size_t j = 0; j < vvpMapPointMatches[i].size(); j++)
        {
            WAIMapPoint* pMP = vvpMapPointMatches[i][j];

            if (pMP && pMP->Observations() > 1)
            {
                const cv::KeyPoint& kp = mCurrentFrame.mvKeys[j];
                vP2D.push_back(kp.pt);
                auto Pos = pMP->worldPosVec();
                vP3Dw.push_back(cv::Point3f(Pos.x, Pos.y, Pos.z));
            }
        }
    }
}

bool WAI::ModeOrbSlam2::relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query WAIKeyFrame Database for keyframe candidates for relocalisation
    vector<WAIKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

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
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            //cout << "Num matches: " << nmatches << endl;
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
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
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<WAIMapPoint*> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<WAIMapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again:
                //ghm1: mappoints seen in the keyframe which was found as candidate via BoW-search are projected into
                //the current frame using the position that was calculated using the matches from BoW matcher
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
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

    if (!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

bool WAI::ModeOrbSlam2::trackReferenceKeyFrame()
{
    //This routine is called if current tracking state is OK but we have NO valid motion model
    //1. Berechnung des BoW-Vectors für den current frame
    //2. using BoW we search mappoint matches (from reference keyframe) with orb in current frame (ORB that belong to the same vocabulary node (at a certain level))
    //3. if there are less than 15 matches return.
    //4. we use the pose found for the last frame as initial pose for the current frame
    //5. This pose is optimized using the matches to map points found by BoW search with reference frame
    //6. Matches classified as outliers by the optimization routine are updated in the mvpMapPoints vector in the current frame and the valid matches are counted
    //7. If there are more than 10 valid matches the reference frame tracking was successful.

    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher           matcher(0.7, true);
    vector<WAIMapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                WAIMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]   = false;
                pMP->mbTrackInView            = false;
                pMP->mnLastFrameSeen          = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    return nmatchesMap >= 10;
}

bool WAI::ModeOrbSlam2::trackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    //(UpdateLocalKeyFrames())
    //1. For all matches to mappoints we search for the keyframes in which theses mappoints have been observed
    //2. We set the keyframe with the most common matches to mappoints as reference keyframe. Simultaniously a list of localKeyFrames is maintained (mvpLocalKeyFrames)
    //(UpdateLocalPoints())
    //3. Pointers to map points are added to mvpLocalMapPoints and the id of the current frame is stored into mappoint instance (mnTrackReferenceForFrame).
    //(SearchLocalPoints())
    //4. The so found local map is searched for additional matches. We check if it is not matched already, if it is in frustum and then the ORBMatcher is used to search feature matches by projection.
    //(ORBMatcher::searchByProjection())
    //5.
    //(this function)
    //6. The Pose is optimized using the found additional matches and the already found pose as initial guess

    updateLocalMap();
    searchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!_onlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    {
                        mnMatchesInliers++;
                    }
                }
                else
                {
                    mnMatchesInliers++;
                }
            }
            //else if (mSensor == System::STEREO)
            //    mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
    {
        //cout << "mnMatchesInliers: " << mnMatchesInliers << endl;
        return false;
    }

    if (mnMatchesInliers < 30)
        return false;
    else
        return true;
}

void WAI::ModeOrbSlam2::updateLocalMap()
{
    // This is for visualization
    //mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    updateLocalKeyFrames();
    updateLocalPoints();
}

void WAI::ModeOrbSlam2::searchLocalPoints()
{
    // Do not search map points already matched
    for (vector<WAIMapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
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
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView   = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<WAIMapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        WAIMapPoint* pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills WAIMapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            //ghm1 test:
            //if (!_image.empty())
            //{
            //    WAIPoint2f ptProj(pMP->mTrackProjX, pMP->mTrackProjY);
            //    cv::rectangle(_image,
            //        cv::Rect(ptProj.x - 3, ptProj.y - 3, 7, 7),
            //        Scalar(0, 0, 255));
            //}
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int        th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void WAI::ModeOrbSlam2::updateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<WAIKeyFrame*, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            WAIMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<WAIKeyFrame*, size_t> observations = pMP->GetObservations();
                for (map<WAIKeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int          max    = 0;
    WAIKeyFrame* pKFmax = static_cast<WAIKeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<WAIKeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        WAIKeyFrame* pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max    = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<WAIKeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        WAIKeyFrame* pKF = *itKF;

        const vector<WAIKeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<WAIKeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            WAIKeyFrame* pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
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
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        WAIKeyFrame* pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF               = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

void WAI::ModeOrbSlam2::updateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (vector<WAIKeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        WAIKeyFrame*               pKF   = *itKF;
        const vector<WAIMapPoint*> vpMPs = pKF->GetMapPointMatches();

        for (vector<WAIMapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            WAIMapPoint* pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

bool WAI::ModeOrbSlam2::trackWithMotionModel()
{
    //This method is called if tracking is OK and we have a valid motion model
    //1. UpdateLastFrame(): ...
    //2. We set an initial pose into current frame, which is the pose of the last frame corrected by the motion model (expected motion since last frame)
    //3. Reinitialization of the assotiated map points to key points in the current frame to NULL
    //4. We search for matches with associated mappoints from lastframe by projection to the current frame. A narrow window is used.
    //5. If we found less than 20 matches we search again as before but in a wider search window.
    //6. If we have still less than 20 matches tracking with motion model was unsuccessful
    //7. Else the pose is Optimized
    //8. Matches classified as outliers by the optimization routine are updated in the mvpMapPoints vector in the current frame and the valid matches are counted
    //9. If less than 10 matches to the local map remain the tracking with visual odometry is activated (mbVO = true) and that means no tracking with motion model or reference keyframe
    //10. The tracking with motion model was successful, if we found more than 20 matches to map points

    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    updateLastFrame();

    //this adds the motion differnce between the last and the before-last frame to the pose of the last frame to estimate the position of the current frame
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<WAIMapPoint*>(NULL));

    // Project points seen in previous frame
    int th       = 15;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, true);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<WAIMapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, true);
    }

    if (nmatches < 20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                WAIMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]   = false;
                pMP->mbTrackInView            = false;
                pMP->mnLastFrameSeen          = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    if (_onlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap >= 10;
}

bool WAI::ModeOrbSlam2::trackWithOptFlow()
{ //parameter of this function:
    int   addThres       = 2;
    float maxReprojError = 10.0;

    if (mLastFrame.mvKeys.size() < 100)
    {
        return false;
    }

    std::vector<uint8_t> status;
    std::vector<float>   err;
    cv::Size             winSize(15, 15);

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                              1,     // terminate after this many iterations, or
                              0.03); // when the search window moves by less than this

    std::vector<cv::Point2f> keyPointCoordinatesLastFrame;
    vector<WAIMapPoint*>     matchedMapPoints;
    vector<cv::KeyPoint>     matchedKeyPoints;

    if (_optFlowOK)
    {
        //last time we successfully tracked with optical flow
        matchedMapPoints = _optFlowMapPtsLastFrame;
        matchedKeyPoints = _optFlowKeyPtsLastFrame;
        for (int i = 0; i < _optFlowKeyPtsLastFrame.size(); i++)
        {
            keyPointCoordinatesLastFrame.push_back(_optFlowKeyPtsLastFrame[i].pt);
        }
    }
    else
    {
        //this is the first run of optical flow after lost state
        for (int i = 0; i < mLastFrame.mvpMapPoints.size(); i++)
        {
            if (mLastFrame.mvpMapPoints[i] && !mLastFrame.mvbOutlier[i])
            {
                keyPointCoordinatesLastFrame.push_back(mLastFrame.mvKeys[i].pt);

                matchedMapPoints.push_back(mLastFrame.mvpMapPoints[i]);
                matchedKeyPoints.push_back(mLastFrame.mvKeys[i]);
            }
        }
    }

    if (!keyPointCoordinatesLastFrame.size())
    {
        return false;
    }

    // Find closest possible feature points based on optical flow
    std::vector<cv::Point2f> pred2DPoints(keyPointCoordinatesLastFrame.size());

    cv::calcOpticalFlowPyrLK(
      mLastFrame.imgGray,           // Previous frame
      mCurrentFrame.imgGray,        // Current frame
      keyPointCoordinatesLastFrame, // Previous and current keypoints coordinates.The latter will be
      pred2DPoints,                 // expanded if more good coordinates are detected during OptFlow
      status,                       // Output vector for keypoint correspondences (1 = match found)
      err,                          // Error size for each flow
      winSize,                      // Search window for each pyramid level
      3,                            // Max levels of pyramid creation
      criteria,                     // Configuration from above
      0,                            // Additional flags
      0.01);                        // Minimal Eigen threshold

    // Only use points which are not wrong in any way during the optical flow calculation
    std::vector<cv::Point2f> frame2DPoints;
    std::vector<cv::Point3f> model3DPoints;
    vector<WAIMapPoint*>     trackedMapPoints;
    vector<cv::KeyPoint>     trackedKeyPoints;

    mnMatchesInliers = 0;

    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            // TODO(jan): if pred2DPoints is really expanded during optflow, then the association
            // to 3D points is maybe broken?
            frame2DPoints.push_back(pred2DPoints[i]);
            cv::Point3f p3(matchedMapPoints[i]->GetWorldPos());
            model3DPoints.push_back(p3);

            matchedKeyPoints[i].pt.x = pred2DPoints[i].x;
            matchedKeyPoints[i].pt.y = pred2DPoints[i].y;

            trackedMapPoints.push_back(matchedMapPoints[i]);
            trackedKeyPoints.push_back(matchedKeyPoints[i]);

            std::lock_guard<std::mutex> guard(_nMapMatchesLock);
            mnMatchesInliers++;
        }
    }

    //todo ghm1:
    //- insert tracked points into grid
    //- update grid with matches from current frames tracked map points
    //- how can we make sure that we do not track the same point multiple times?
    //  -> we know the pointer to the mappoints and only add a new tracking points whose mappoint is not in a gridcell yet
    //- we dont want to track too many points, so we prefer points with the most observations
    _optFlowGridElementWidthInv  = static_cast<float>(OPTFLOW_GRID_COLS) / static_cast<float>(WAIFrame::mnMaxX - WAIFrame::mnMinX);
    _optFlowGridElementHeightInv = static_cast<float>(OPTFLOW_GRID_ROWS) / static_cast<float>(WAIFrame::mnMaxY - WAIFrame::mnMinY);
    std::vector<std::size_t> gridOptFlow[OPTFLOW_GRID_COLS][OPTFLOW_GRID_ROWS];
    std::vector<std::size_t> gridCurrFrame[OPTFLOW_GRID_COLS][OPTFLOW_GRID_ROWS];
    //insert optical flow points into grid
    for (int i = 0; i < trackedKeyPoints.size(); ++i)
    {
        int nGridPosX, nGridPosY;
        if (posInGrid(trackedKeyPoints[i], nGridPosX, nGridPosY, WAIFrame::mnMinX, WAIFrame::mnMinY))
            gridOptFlow[nGridPosX][nGridPosY].push_back(i);
    }
    //insert current frame points into grid
    for (int i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i] != NULL)
        {
            int nGridPosX, nGridPosY;
            if (posInGrid(mCurrentFrame.mvKeys[i], nGridPosX, nGridPosY, WAIFrame::mnMinX, WAIFrame::mnMinY))
                gridCurrFrame[nGridPosX][nGridPosY].push_back(i);
        }
    }

    //try to add tracking points from gridCurrFrame to trackedMapPoints and trackedKeyPoints where missing in gridOptFlow
    for (int i = 0; i < OPTFLOW_GRID_COLS; i++)
    {
        for (int j = 0; j < OPTFLOW_GRID_ROWS; j++)
        {
            const auto& optFlowCell = gridOptFlow[i][j];
            if (optFlowCell.size() < addThres)
            {
                const std::vector<size_t>& indices = gridCurrFrame[i][j];
                for (auto indexCF : indices)
                {
                    const cv::KeyPoint& keyPt = mCurrentFrame.mvKeys[indexCF];
                    WAIMapPoint*        mapPt = mCurrentFrame.mvpMapPoints[indexCF];
                    if (mapPt)
                    {
                        //check that this map point is not already referenced in this cell of gridOptFlow
                        bool alreadyContained = false;
                        for (auto indexOF : optFlowCell)
                        {
                            if (trackedMapPoints[indexOF] == mapPt)
                            {
                                alreadyContained = true;
                                break;
                            }
                        }

                        if (!alreadyContained)
                        {
                            //add to tracking set of mappoints and keypoints
                            trackedKeyPoints.push_back(keyPt);
                            trackedMapPoints.push_back(mapPt);
                            frame2DPoints.push_back(keyPt.pt);
                            model3DPoints.push_back(cv::Point3f(mapPt->GetWorldPos()));
                        }
                    }
                }
            }
        }
    }

    if (trackedKeyPoints.size() < matchedKeyPoints.size() * 0.75)
    {
        return false;
    }

    /////////////////////
    // Pose Estimation //
    /////////////////////

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat om   = mCurrentFrame.mTcw;
    cv::Rodrigues(om.rowRange(0, 3).colRange(0, 3), rvec);
    tvec = om.colRange(3, 4).rowRange(0, 3);

    bool foundPose = cv::solvePnP(model3DPoints,
                                  frame2DPoints,
                                  _camera->getCameraMatrix(),
                                  _camera->getDistortionMatrix(),
                                  rvec,
                                  tvec,
                                  true);

    if (foundPose)
    {
        cv::Mat Tcw         = cv::Mat::eye(4, 4, CV_32F);
        Tcw.at<float>(0, 3) = tvec.at<float>(0, 0);
        Tcw.at<float>(1, 3) = tvec.at<float>(1, 0);
        Tcw.at<float>(2, 3) = tvec.at<float>(2, 0);

        cv::Mat Rcw = cv::Mat::zeros(3, 3, CV_32F);
        cv::Rodrigues(rvec, Rcw);

        Tcw.at<float>(0, 0) = Rcw.at<float>(0, 0);
        Tcw.at<float>(1, 0) = Rcw.at<float>(1, 0);
        Tcw.at<float>(2, 0) = Rcw.at<float>(2, 0);
        Tcw.at<float>(0, 1) = Rcw.at<float>(0, 1);
        Tcw.at<float>(1, 1) = Rcw.at<float>(1, 1);
        Tcw.at<float>(2, 1) = Rcw.at<float>(2, 1);
        Tcw.at<float>(0, 2) = Rcw.at<float>(0, 2);
        Tcw.at<float>(1, 2) = Rcw.at<float>(1, 2);
        Tcw.at<float>(2, 2) = Rcw.at<float>(2, 2);
        _optFlowTcw         = Tcw;

        //remove points with bad reprojection error:
        //project mappoints onto image plane
        std::vector<cv::Point2f> projectedPts;
        cv::projectPoints(model3DPoints,
                          rvec,
                          tvec,
                          _camera->getCameraMatrix(),
                          _camera->getDistortionMatrix(),
                          projectedPts);

        _optFlowMapPtsLastFrame.clear();
        _optFlowKeyPtsLastFrame.clear();
        for (int i = 0; i < trackedMapPoints.size(); ++i)
        {
            //calculate reprojection error
            float error = cv::norm(cv::Mat(projectedPts[i]), cv::Mat(frame2DPoints[i]));
            if (error < maxReprojError)
            {
                _optFlowMapPtsLastFrame.push_back(trackedMapPoints[i]);
                _optFlowKeyPtsLastFrame.push_back(trackedKeyPoints[i]);
            }
        }
        //trackingType = TrackingType_OptFlow;
    }

    return foundPose;
}

void WAI::ModeOrbSlam2::updateLastFrame()
{
    // Update pose according to reference keyframe
    WAIKeyFrame* pRef = mLastFrame.mpReferenceKF;
    //cout << "pRef pose: " << pRef->GetPose() << endl;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    //GHM1:
    //l = last, w = world, r = reference
    //Tlr is the relative transformation for the last frame wrt to reference frame
    //(because the relative pose for the current frame is added at the end of tracking)
    //Refer last frame pose to world: Tlw = Tlr * Trw
    //So it seems, that the frames pose does not always refer to world frame...?
    mLastFrame.SetPose(Tlr * pRef->GetPose());
}

void WAI::ModeOrbSlam2::globalBundleAdjustment()
{
    Optimizer::GlobalBundleAdjustemnt(_map, 20);
    //_mapNode->updateAll(*_map);
}

#if 0
size_t WAI::ModeOrbSlam2::getSizeOf()
{
    size_t size = 0;

    size += sizeof(*this);
    //add size of local mapping
    //add size of loop closing
    //add size of

    return size;
}
#endif

WAIKeyFrame* WAI::ModeOrbSlam2::currentKeyFrame()
{
    WAIKeyFrame* result = mCurrentFrame.mpReferenceKF;

    return result;
}

void WAI::ModeOrbSlam2::decorate()
{
    //calculation of mean reprojection error of all matches
    calculateMeanReprojectionError();
    //calculate pose difference
    calculatePoseDifference();
    //show rectangle for key points in video that where matched to map points
    cv::Mat imgRGB = _camera->getImageRGB();
    decorateVideoWithKeyPoints(imgRGB);
    decorateVideoWithKeyPointMatches(imgRGB);
    //decorate scene with matched map points, local map points and matched map points
    //decorateScene();
}

void WAI::ModeOrbSlam2::calculateMeanReprojectionError()
{
    //calculation of mean reprojection error
    double reprojectionError = 0.0;
    int    n                 = 0;

    //current frame extrinsic
    const cv::Mat Rcw = mCurrentFrame.GetRotationCW();
    const cv::Mat tcw = mCurrentFrame.GetTranslationCW();
    //current frame intrinsics
    const float fx = mCurrentFrame.fx;
    const float fy = mCurrentFrame.fy;
    const float cx = mCurrentFrame.cx;
    const float cy = mCurrentFrame.cy;

    for (size_t i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                {
                    // 3D in absolute coordinates
                    cv::Mat Pw = mCurrentFrame.mvpMapPoints[i]->GetWorldPos();
                    // 3D in camera coordinates
                    const cv::Mat Pc  = Rcw * Pw + tcw;
                    const float&  PcX = Pc.at<float>(0);
                    const float&  PcY = Pc.at<float>(1);
                    const float&  PcZ = Pc.at<float>(2);

                    // Check positive depth
                    if (PcZ < 0.0f)
                        continue;

                    // Project in image and check it is not outside
                    const float invz = 1.0f / PcZ;
                    const float u    = fx * PcX * invz + cx;
                    const float v    = fy * PcY * invz + cy;

                    cv::Point2f ptProj(u, v);
                    //Use distorted points because we have to undistort the image later
                    const auto& ptImg = mCurrentFrame.mvKeysUn[i].pt;

                    ////draw projected point
                    //cv::rectangle(image,
                    //    cv::Rect(ptProj.x - 3, ptProj.y - 3, 7, 7),
                    //    Scalar(255, 0, 0));

                    reprojectionError += cv::norm(cv::Mat(ptImg), cv::Mat(ptProj));
                    n++;
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(_meanProjErrorLock);
        if (n > 0)
        {
            _meanReprojectionError = reprojectionError / n;
        }
        else
        {
            _meanReprojectionError = -1;
        }
    }
}

void WAI::ModeOrbSlam2::calculatePoseDifference()
{
    std::lock_guard<std::mutex> guard(_poseDiffLock);
    //calculation of L2 norm of the difference between the last and the current camera pose
    if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
        _poseDifference = norm(mLastFrame.mTcw - mCurrentFrame.mTcw);
    else
        _poseDifference = -1.0;
}

void WAI::ModeOrbSlam2::decorateVideoWithKeyPoints(cv::Mat& image)
{
    //show rectangle for all keypoints in current image
    if (_showKeyPoints)
    {
        for (size_t i = 0; i < mCurrentFrame.N; i++)
        {
            //Use distorted points because we have to undistort the image later
            //const auto& pt = mCurrentFrame.mvKeys[i].pt;
            const auto& pt = mCurrentFrame.mvKeys[i].pt;
            cv::rectangle(image,
                          cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                          cv::Scalar(0, 0, 255));
        }
    }
}

void WAI::ModeOrbSlam2::decorateVideoWithKeyPointMatches(cv::Mat& image)
{
    //show rectangle for key points in video that where matched to map points
    if (_showKeyPointsMatched)
    {
        if (_optFlowOK)
        {
            for (size_t i = 0; i < _optFlowKeyPtsLastFrame.size(); i++)
            {
                //Use distorted points because we have to undistort the image later
                const auto& pt = _optFlowKeyPtsLastFrame[i].pt;
                cv::rectangle(image,
                              cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                              cv::Scalar(0, 255, 0));
            }
        }
        else
        {
            for (size_t i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i])
                {
                    if (!mCurrentFrame.mvbOutlier[i])
                    {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        {
                            //Use distorted points because we have to undistort the image later
                            const auto& pt = mCurrentFrame.mvKeys[i].pt;
                            cv::rectangle(image,
                                          cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                                          cv::Scalar(0, 255, 0));
                        }
                    }
                }
            }
        }
    }
}

void WAI::ModeOrbSlam2::loadMapData(std::vector<WAIKeyFrame*> keyFrames,
                                    std::vector<WAIMapPoint*> mapPoints,
                                    int                       numLoopClosings)
{
    for (WAIKeyFrame* kf : keyFrames)
    {
        _map->AddKeyFrame(kf);

        //Update keyframe database:
        //add to keyframe database
        mpKeyFrameDatabase->add(kf);
    }

    for (WAIMapPoint* point : mapPoints)
    {
        _map->AddMapPoint(point);
    }

    _map->setNumLoopClosings(numLoopClosings);

    setInitialized(true);
}
