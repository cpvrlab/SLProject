#include <WAIModeOrbSlam2.h>
#include <AverageTiming.h>
#include <Utils.h>

WAI::ModeOrbSlam2::ModeOrbSlam2(ORB_SLAM2::KPextractor* kpExtractor,
                                ORB_SLAM2::KPextractor* kpIniExtractor,
                                cv::Mat                 cameraMat,
                                cv::Mat                 distortionMat,
                                const Params&           params,
                                std::string             orbVocFile,
                                bool                    applyMinAccScoreFilter)
  : mpExtractor(kpExtractor),
    mpIniExtractor(kpIniExtractor),
    _params(params),
    _applyMinAccScoreFilter(applyMinAccScoreFilter)
{
    //we have to reset global static stuff here
    WAIKeyFrame::nNextId = 0; //will be updated when a map is loaded
    WAIFrame::nNextId    = 0;
    WAIMapPoint::nNextId = 0;
    // Tell WAIFrame to compute image bounds on first instantiation
    WAIFrame::mbInitialComputations = true;

    cameraMat.convertTo(_cameraMat, CV_32F);
    distortionMat.convertTo(_distortionMat, CV_32F);

    //load visual vocabulary for relocalization
    if (!WAIOrbVocabulary::initialize(orbVocFile))
        throw std::runtime_error("ModeOrbSlam2: could not find vocabulary file: " + orbVocFile);
    mpVocabulary = WAIOrbVocabulary::get();

    //instantiate and load slam map
    mpKeyFrameDatabase = new WAIKeyFrameDB(*mpVocabulary);

    _map = new WAIMap("Map");

    if (_map->KeyFramesInMap())
        _initialized = true;
    else
        _initialized = false;

    //instantiate local mapping
    mpLocalMapper = new ORB_SLAM2::LocalMapping(_map, 1, mpVocabulary, _params.cullRedundantPerc);
    mpLoopCloser  = new ORB_SLAM2::LoopClosing(_map, mpKeyFrameDatabase, mpVocabulary, false, false);

    mpLocalMapper->SetLoopCloser(mpLoopCloser);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    if (!_params.serial)
    {
        mptLocalMapping = new std::thread(&LocalMapping::Run, mpLocalMapper);
        //if (!_params.fixOldKfs)
        mptLoopClosing = new std::thread(&LoopClosing::Run, mpLoopCloser);
    }

    _state = TrackingState_Initializing;
    _pose  = cv::Mat(4, 4, CV_32F);
}

WAI::ModeOrbSlam2::ModeOrbSlam2(ORB_SLAM2::KPextractor* kpExtractor,
                                ORB_SLAM2::KPextractor* kpIniExtractor,
                                ORB_SLAM2::KPextractor* kpMarkerExtractor,
                                std::string             markerFile,
                                cv::Mat                 cameraMat,
                                cv::Mat                 distortionMat,
                                const Params&           params,
                                std::string             orbVocFile,
                                bool                    applyMinAccScoreFilter)
  : mpExtractor(kpExtractor),
    mpIniExtractor(kpIniExtractor),
    _markerExtractor(kpMarkerExtractor),
    _params(params),
    _applyMinAccScoreFilter(applyMinAccScoreFilter)
{
    //we have to reset global static stuff here
    WAIKeyFrame::nNextId = 0; //will be updated when a map is loaded
    WAIFrame::nNextId    = 0;
    WAIMapPoint::nNextId = 0;
    // Tell WAIFrame to compute image bounds on first instantiation
    WAIFrame::mbInitialComputations = true;

    cameraMat.convertTo(_cameraMat, CV_32F);
    distortionMat.convertTo(_distortionMat, CV_32F);

    //load visual vocabulary for relocalization
    if (!WAIOrbVocabulary::initialize(orbVocFile))
        throw std::runtime_error("ModeOrbSlam2: could not find vocabulary file: " + orbVocFile);
    mpVocabulary = WAIOrbVocabulary::get();

    //instantiate and load slam map
    mpKeyFrameDatabase = new WAIKeyFrameDB(*mpVocabulary);

    _map = new WAIMap("Map");

    if (_map->KeyFramesInMap())
        _initialized = true;
    else
        _initialized = false;

    //instantiate local mapping
    mpLocalMapper = new ORB_SLAM2::LocalMapping(_map, 1, mpVocabulary, _params.cullRedundantPerc);
    mpLoopCloser  = new ORB_SLAM2::LoopClosing(_map, mpKeyFrameDatabase, mpVocabulary, false, false);

    mpLocalMapper->SetLoopCloser(mpLoopCloser);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    if (!_params.serial)
    {
        mptLocalMapping = new std::thread(&LocalMapping::Run, mpLocalMapper);
        //if (!_params.fixOldKfs)
        mptLoopClosing = new std::thread(&LoopClosing::Run, mpLoopCloser);
    }

    _state = TrackingState_Initializing;
    _pose  = cv::Mat(4, 4, CV_32F);

    if (_markerExtractor && !markerFile.empty())
        _markerFrame = createMarkerFrame(markerFile, _markerExtractor);
}

void WAI::ModeOrbSlam2::setVocabulary(std::string orbVocFile)
{
    requestStateIdle();
    WAIOrbVocabulary::free();
    WAIOrbVocabulary::initialize(orbVocFile);
    mpVocabulary = WAIOrbVocabulary::get();
    mpKeyFrameDatabase->changeVocabulary(*mpVocabulary, getKeyFrames());
    if (mpLocalMapper)
        mpLocalMapper->SetVocabulary(mpVocabulary);
    if (mpLoopCloser)
        mpLoopCloser->SetVocabulary(mpVocabulary);
    resume();
}

WAI::ModeOrbSlam2::~ModeOrbSlam2()
{
    if (!_params.serial)
    {
        mpLocalMapper->RequestFinish();
        //if (!_params.fixOldKfs)
        mpLoopCloser->RequestFinish();

        // Wait until all thread have effectively stopped
        mptLocalMapping->join();
        if (mptLoopClosing)
            mptLoopClosing->join();
    }

    if (mpLocalMapper) delete mpLocalMapper;
    if (mpLoopCloser) delete mpLoopCloser;
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

bool WAI::ModeOrbSlam2::update(cv::Mat& imageGray, cv::Mat& imageRGB)
{
    _poseSet = false;
    stateTransition();

    switch (_state)
    {
        case TrackingState_Initializing:
        {
            initialize(imageGray, imageRGB);
        }
        break;

        case TrackingState_TrackingOK:
        case TrackingState_TrackingLost:
        {
            //relocalize or track 3d points
            track3DPts(imageGray, imageRGB);
        }
        break;

        case TrackingState_Idle:
        case TrackingState_None:
        default:
        {
        }
        break;
    }

    return _poseSet;
}

int WAI::ModeOrbSlam2::getMapPointCount()
{
    int result = _map->MapPointsInMap();

    return result;
}

int WAI::ModeOrbSlam2::getMapPointMatchesCount()
{
    int result = mnMatchesInliers;

    return result;
}

int WAI::ModeOrbSlam2::getKeyFrameCount()
{
    int result = _map->KeyFramesInMap();

    return result;
}

std::string WAI::ModeOrbSlam2::getLoopCloseStatus()
{
    std::string result = mpLoopCloser->getStatusString();

    return result;
}

int WAI::ModeOrbSlam2::getLoopCloseCount()
{
    int result = _map->getNumLoopClosings();

    return result;
}

int WAI::ModeOrbSlam2::getKeyFramesInLoopCloseQueueCount()
{
    int result = mpLoopCloser->numOfKfsInQueue();

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

std::vector<WAIMapPoint*> WAI::ModeOrbSlam2::getMarkerCornerMapPoints()
{
    std::vector<WAIMapPoint*> result;

    if (_mpUL)
    {
        if (_mpUL->isBad())
        {
            Utils::log("WAI", "_mpUL->isBad()!!!");
        }
        result.push_back(_mpUL);
    }
    if (_mpUR)
    {
        if (_mpUR->isBad())
        {
            Utils::log("WAI", "_mpUR->isBad()!!!");
        }
        result.push_back(_mpUR);
    }
    if (_mpLL)
    {
        if (_mpLL->isBad())
        {
            Utils::log("WAI", "_mpLL->isBad()!!!");
        }
        result.push_back(_mpLL);
    }
    if (_mpLR)
    {
        if (_mpLR->isBad())
        {
            Utils::log("WAI", "_mpLR->isBad()!!!");
        }
        result.push_back(_mpLR);
    }

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
                if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    result.push_back(mCurrentFrame.mvpMapPoints[i]);
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
    std::vector<WAIMapPoint*>   result = mvpLocalMapPoints;

    return result;
}

std::vector<WAIKeyFrame*> WAI::ModeOrbSlam2::getKeyFrames()
{
    std::lock_guard<std::mutex> guard(_mapLock);
    std::vector<WAIKeyFrame*>   result = _map->GetAllKeyFrames();

    return result;
}

bool WAI::ModeOrbSlam2::getTrackOptFlow()
{
    std::lock_guard<std::mutex> guard(_optFlowLock);
    return _params.trackOptFlow;
}

void WAI::ModeOrbSlam2::setTrackOptFlow(bool flag)
{
    std::lock_guard<std::mutex> guard(_optFlowLock);
    _params.trackOptFlow = flag;
    _optFlowOK           = false;
}

void WAI::ModeOrbSlam2::disableMapping()
{
    _params.onlyTracking = true;
    if (!_params.serial)
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
    _params.onlyTracking = false;
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

void WAI::ModeOrbSlam2::initialize(cv::Mat& imageGray, cv::Mat& imageRGB)
{
    //1. if there are more than 100 keypoints in the current frame, the Initializer is instantiated
    //2. if there are less than 100 keypoints in the next frame, the Initializer is deinstantiated again
    //3. else if there are more than 100 keypoints we try to match the keypoints in the current with the initial frame
    //4. if we found less than 100 matches between the current and the initial keypoints, the Initializer is deinstantiated
    //5. else we try to initialize: that means a homograhy and a fundamental matrix are calculated in parallel and 3D points are triangulated initially
    //6. if the initialization (by homograhy or fundamental matrix) was successful an inital map is created:
    //  - two keyframes are generated from the initial and the current frame and added to keyframe database and map
    //  - a mappoint is instantiated from the triangulated 3D points and all necessary members are calculated (distinctive descriptor, depth and normal, add observation reference of keyframes)
    //  - a global bundle adjustment is applied
    //  - the two new keyframes are added to the local mapper and the local mapper is started twice
    //  - the tracking state is changed to TRACKING/INITIALIZED

    // NOTE(dgj1): was originally 100
    int matchesNeeded = 100;

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(_map->mMutexMapUpdate, std::defer_lock);
    if (!_params.serial)
    {
        lock.lock();
    }

    mCurrentFrame = WAIFrame(imageGray,
                             0.0,
                             mpIniExtractor,
                             _cameraMat,
                             _distortionMat,
                             mpVocabulary,
                             _params.retainImg);

    if (!mpInitializer)
    {
// NOTE(dgj1): find matches with marker if necessary
// We only initialize if the first frame has enough matches to the marker
// TODO(dgj1): rethink this constraint
#if 0
        if (_createMarkerMap)
        {
            ORBmatcher               matcher(0.9, true);
            std::vector<cv::Point2f> prevMatched(_markerFrame.mvKeysUn.size());
            for (size_t i = 0; i < _markerFrame.mvKeysUn.size(); i++)
                prevMatched[i] = _markerFrame.mvKeysUn[i].pt;

            std::vector<int> markerMatchesToCurrentFrame;
            int              nmatches = matcher.SearchForInitialization(_markerFrame, mCurrentFrame, prevMatched, markerMatchesToCurrentFrame, 100);

            if (nmatches <= 30)
            {
                return;
            }
        }
#endif

        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > matchesNeeded)
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
        if ((int)mCurrentFrame.mvKeys.size() <= matchesNeeded)
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
        if (nmatches < matchesNeeded)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        for (unsigned int i = 0; i < mInitialFrame.mvKeys.size(); i++)
        {
            cv::rectangle(imageRGB,
                          mInitialFrame.mvKeys[i].pt,
                          cv::Point(mInitialFrame.mvKeys[i].pt.x + 3, mInitialFrame.mvKeys[i].pt.y + 3),
                          cv::Scalar(0, 0, 255));
        }

        //ghm1: decorate image with tracked matches
        for (unsigned int i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] >= 0)
            {
                cv::line(imageRGB,
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
            //cv::imwrite("mCurrentFrame.png", mCurrentFrame.imgGray);
            //cv::imwrite("mInitialFrame.png", mInitialFrame.imgGray);

            //std::cout << "Initialize num matches: " << mvIniMatches.size() << std::endl;
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            //std::cout << "ModeOrbSlam2::initialize" << std::endl;
            //std::cout << "R: " << Rcw << std::endl;
            //std::cout << "t: " << tcw << std::endl;

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            bool mapInitializedSuccessfully = createInitialMapMonocular(matchesNeeded);

            if (mapInitializedSuccessfully)
            {
                //mark tracking as initialized
                _initialized = true;
                _bOK         = true;
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

void WAI::ModeOrbSlam2::track3DPts(cv::Mat& imageGray, cv::Mat& imageRGB)
{
    mCurrentFrame = WAIFrame(imageGray,
                             0.0,
                             mpExtractor,
                             _cameraMat,
                             _distortionMat,
                             mpVocabulary,
                             _params.retainImg);

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(_map->mMutexMapUpdate, std::defer_lock);
    if (!_params.serial)
    {
        lock.lock();
    }

    //mLastProcessedState = mState;
    //bool bOK;
    _bOK = false;
    //trackingType = TrackingType_None;

    if (!_params.onlyTracking)
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
            _bOK = relocalization(mCurrentFrame, mpKeyFrameDatabase, &mnLastRelocFrameId, *_map, _applyMinAccScoreFilter, false);
        }
    }
    else
    {
        // Localization Mode: Local Mapping is deactivated
        if (_state == TrackingState_TrackingLost)
        {
            _bOK       = relocalization(mCurrentFrame, mpKeyFrameDatabase, &mnLastRelocFrameId, *_map, _applyMinAccScoreFilter);
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
                    TcwMM   = mCurrentFrame.mTcw.clone();
                }
                bOKReloc = relocalization(mCurrentFrame, mpKeyFrameDatabase, &mnLastRelocFrameId, *_map, _applyMinAccScoreFilter);
                //relocalization method is not valid but the velocity model method
                if (bOKMM && !bOKReloc)
                {
                    mCurrentFrame.SetPose(TcwMM);
                    mCurrentFrame.mvpMapPoints = vpMPsMM;

                    if (mbVO)
                    {
                        for (int i = 0; i < mCurrentFrame.N; i++)
                        {
                            //if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                            if (mCurrentFrame.mvpMapPoints[i])
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
    if (!_params.onlyTracking)
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

    if (_params.trackOptFlow && _bOK && _state == TrackingState_TrackingOK)
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

            _pose    = Tcw;
            _poseSet = true;
        }

        // Clean VO matches
        /*
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
        */

        //ghm1: manual local mapping of current frame
        if (needNewKeyFrame())
        {
            createNewKeyFrame();

            if (_params.serial)
            {
                //call local mapper
                mpLocalMapper->RunOnce();
                //normally the loop closing would feed the keyframe database, but we do it here
                //mpKeyFrameDatabase->add(mpLastKeyFrame);

                //loop closing
                //if (!_params.fixOldKfs)
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
        /*
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
            }
        }
        */
    }

    if (!mCurrentFrame.mpReferenceKF)
    {
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    decorate(imageRGB);

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

bool WAI::ModeOrbSlam2::createInitialMapMonocular(int mapPointsNeeded)
{
    //ghm1: reset nNextId to 0! This is important otherwise the first keyframe cannot be identified via its id and a lot of stuff gets messed up!
    //One problem we identified is in UpdateConnections: the first one is not allowed to have a parent,
    //because the second will set the first as a parent too. We get problems later during culling.
    //This also fixes a problem in first GlobalBundleAdjustment which messed up the map after a reset.
    WAIKeyFrame::nNextId = 0;

    // Create KeyFrames
    WAIKeyFrame* pKFini = new WAIKeyFrame(mInitialFrame, mpKeyFrameDatabase);
    WAIKeyFrame* pKFcur = new WAIKeyFrame(mCurrentFrame, mpKeyFrameDatabase);

    pKFini->ComputeBoW(mpVocabulary);
    pKFcur->ComputeBoW(mpVocabulary);

    // Insert KFs in the map
    _map->AddKeyFrame(pKFini);
    _map->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        WAIMapPoint* pMP = new WAIMapPoint(worldPos, pKFcur);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;

        //Add to Map
        _map->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    //cout << "New Map created with " << _map->MapPointsInMap() << " points" << endl;

    // Bundle Adjustment
    Optimizer::GlobalBundleAdjustemnt(_map, 20);

    // Set median depth to 1
    float medianDepth    = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < mapPointsNeeded)
    {
        Utils::log("WAI", "Wrong initialization, reseting...");
        reset();
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

    //ghm1: run local mapping once
    if (_params.serial)
    {
        mpLocalMapper->RunOnce();
        //todo: why two times??
        mpLocalMapper->RunOnce();
    }

    // Bundle Adjustment
    //Utils::log("WAI","Number of Map points after local mapping: %i", _map->MapPointsInMap());

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
    if (_params.onlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = _map->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // unless less than 30 keyframes are in the map
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
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            return false;
        }
    }
    else
    {
        return false;
    }
}

void WAI::ModeOrbSlam2::createNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    WAIKeyFrame* pKF = new WAIKeyFrame(mCurrentFrame, mpKeyFrameDatabase);

    mpReferenceKF               = pKF;
    mCurrentFrame.mpReferenceKF = pKF;
    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame   = pKF;
}

void WAI::ModeOrbSlam2::reset()
{
    Utils::log("WAI", "System Reseting");

    // Reset Local Mapping
    if (!_params.serial)
    {
        mpLocalMapper->RequestReset();
    }
    else
    {
        mpLocalMapper->reset();
    }

    //// Reset Loop Closing
    if (!_params.serial)
    {
        //if (!_params.fixOldKfs)
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

    WAIKeyFrame::nNextId            = 0;
    WAIFrame::nNextId               = 0;
    WAIFrame::mbInitialComputations = true;
    WAIMapPoint::nNextId            = 0;

    _bOK         = false;
    _initialized = false;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(nullptr);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    mCurrentFrame  = WAIFrame();
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
    if (!_params.serial)
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
    if (!_params.serial)
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

bool WAI::ModeOrbSlam2::relocalization(WAIFrame&      currentFrame,
                                       WAIKeyFrameDB* keyFrameDB,
                                       unsigned int*  lastRelocFrameId,
                                       WAIMap&        waiMap,
                                       bool           applyMinAccScoreFilter,
                                       bool           relocWithAllKFs)
{
    AVERAGE_TIMING_START("Relocalization");
    // Compute Bag of Words Vector
    currentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query WAIKeyFrame Database for keyframe candidates for relocalisation
    vector<WAIKeyFrame*> vpCandidateKFs;
    if (relocWithAllKFs)
    {
        //alternative candidate search (test code)
        vpCandidateKFs = waiMap.GetAllKeyFrames();
    }
    else
    {
        //TODO(luc): test with 2nd argument to true
        vpCandidateKFs = keyFrameDB->DetectRelocalizationCandidates(&currentFrame, applyMinAccScoreFilter);
    }

    if (!vpCandidateKFs.size() && waiMap.KeyFramesInMap() < 5)
    {
        vpCandidateKFs = waiMap.GetAllKeyFrames();
    }

    //std::cout << "N after DetectRelocalizationCandidates: " << vpCandidateKFs.size() << std::endl;

    if (vpCandidateKFs.empty())
    {
        AVERAGE_TIMING_STOP("Relocalization");
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
            vector<bool> vbOutliers;
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

                int nGood = Optimizer::PoseOptimization(&currentFrame, vbOutliers);

                if (nGood < 10)
                    continue;

                /*
                for (int io = 0; io < currentFrame.N; io++)
                    if (currentFrame.mvbOutlier[io])
                        currentFrame.mvpMapPoints[io] = static_cast<WAIMapPoint*>(NULL);
                */

                // If few inliers, search by projection in a coarse window and optimize again:
                //ghm1: mappoints seen in the keyframe which was found as candidate via BoW-search are projected into
                //the current frame using the position that was calculated using the matches from BoW matcher
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(currentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&currentFrame, vbOutliers);

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

    AVERAGE_TIMING_STOP("Relocalization");
    if (!bMatch)
    {
        return false;
    }
    else
    {
        *lastRelocFrameId = currentFrame.mnId;
        return true;
    }
}

WAIFrame WAI::ModeOrbSlam2::getCurrentFrame()
{
    return mCurrentFrame;
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

    AVERAGE_TIMING_START("TrackReferenceKeyFrame");
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher           matcher(0.7, true);
    vector<WAIMapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        AVERAGE_TIMING_STOP("TrackReferenceKeyFrame");
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    nmatches = Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    /*
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
    */

    AVERAGE_TIMING_STOP("TrackReferenceKeyFrame");
    //return nmatchesMap >= 10;
    return nmatches >= 10;
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

    AVERAGE_TIMING_START("trackLocalMap");
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
            mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
            if (!_params.onlyTracking)
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
            //else if (mSensor == System::STEREO)
            //    mCurrentFrame.mvpMapPoints[i] = static_cast<WAIMapPoint*>(NULL);
        }
    }

    AVERAGE_TIMING_STOP("trackLocalMap");
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
    for (auto itKF = mvpLocalKeyFrames.begin(); itKF != mvpLocalKeyFrames.end(); itKF++)
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
    AVERAGE_TIMING_START("TrackWithMotionModel");
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
    {
        AVERAGE_TIMING_STOP("TrackWithMotionModel");
        return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
            /*
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
                */
        }
    }

    AVERAGE_TIMING_STOP("TrackWithMotionModel");
    if (_params.onlyTracking)
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
            //if (mLastFrame.mvpMapPoints[i] && !mvbOutlier[i])
            if (mLastFrame.mvpMapPoints[i])
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
                                  _cameraMat,
                                  _distortionMat,
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
                          _cameraMat,
                          _distortionMat,
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

//void WAI::ModeOrbSlam2::globalBundleAdjustment()
//{
//    Optimizer::GlobalBundleAdjustemnt(_map, 20);
//    //_mapNode->updateAll(*_map);
//}

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

void WAI::ModeOrbSlam2::decorate(cv::Mat& image)
{
    //calculation of mean reprojection error of all matches
    calculateMeanReprojectionError();
    //calculate pose difference
    calculatePoseDifference();
    //decorateVideoWithKeyPoints(image);
    //decorateVideoWithKeyPointMatches(image);
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

void WAI::ModeOrbSlam2::decorateVideoWithKeyPointMatches(cv::Mat& image)
{
    //show rectangle for key points in video that where matched to map points
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

bool WAI::ModeOrbSlam2::findMarkerHomography(WAIFrame&    markerFrame,
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

WAIFrame WAI::ModeOrbSlam2::createMarkerFrame(std::string markerFile, KPextractor* markerExtractor)
{
    cv::Mat markerImgGray = cv::imread(markerFile, cv::IMREAD_GRAYSCALE);

    float fyCam = _cameraMat.at<float>(1, 1);
    float cyCam = _cameraMat.at<float>(1, 2);
    float fov   = 2.0f * atan2(cyCam, fyCam) * 180.0f / M_PI;

    float cx = (float)markerImgGray.cols * 0.5f;
    float cy = (float)markerImgGray.rows * 0.5f;
    float fy = cy / tanf(fov * 0.5f * M_PI / 180.0);
    float fx = fy;

    // TODO(dgj1): pass actual calibration for marker frame?
    cv::Mat markerCameraMat     = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat markerDistortionMat = cv::Mat::zeros(4, 1, CV_32F);

    WAIFrame result = WAIFrame(markerImgGray, 0.0f, markerExtractor, markerCameraMat, markerDistortionMat, mpVocabulary, true);
    result          = WAIFrame(markerImgGray, 0.0f, markerExtractor, markerCameraMat, markerDistortionMat, mpVocabulary, true);
    result          = WAIFrame(markerImgGray, 0.0f, markerExtractor, markerCameraMat, markerDistortionMat, mpVocabulary, true);
    return result;
}

bool WAI::ModeOrbSlam2::doMarkerMapPreprocessing(std::string markerFile,
                                                 cv::Mat&    nodeTransform,
                                                 float       markerWidthInM)
{
    // Additional steps to save marker map
    // 1. Find matches to marker on two keyframes
    // 1.a Extract features from marker image
    WAIFrame markerFrame = createMarkerFrame(markerFile, _markerExtractor);

    // 1.b Find keyframes with enough matches to marker image
    std::vector<WAIKeyFrame*> kfs = _map->GetAllKeyFrames();

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
    std::vector<WAIMapPoint*> mapPoints = _map->GetAllMapPoints();

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

int WAI::ModeOrbSlam2::getKeyPointCount()
{
    int result = mCurrentFrame.N;

    return result;
}
