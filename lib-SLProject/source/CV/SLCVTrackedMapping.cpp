//#############################################################################
//  File:      SLCVTrackedAruco.cpp
//  Author:    Michael Goettlicher, Marcus Hudritsch
//  Date:      Winter 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers

/*
The OpenCV library version 3.1 with extra module must be present.
If the application captures the live video stream with OpenCV you have
to define in addition the constant SL_USES_CVCAPTURE.
All classes that use OpenCV begin with SLCV.
See also the class docs for SLCVCapture, SLCVCalibration and SLCVTracked
for a good top down information.
*/
#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLCVTrackedMapping.h>
#include <SLCVKeyFrameDB.h>
#include <SLCVMapNode.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <SLCVMapIO.h>
#include <SLCVMapStorage.h>
#include <SLCVOrbVocabulary.h>

#ifndef _WINDOWS
#include <unistd.h>
#endif

using namespace cv;

//-----------------------------------------------------------------------------
SLCVTrackedMapping::SLCVTrackedMapping(SLNode* node,
                                       bool onlyTracking,
                                       SLCVMapNode* mapNode,
                                       bool serial)
    : SLCVTracked(node),
      mbOnlyTracking(onlyTracking),
      SLCVMapTracking(mapNode, true),
      _serial(serial)
{
    //load visual vocabulary for relocalization
    mpVocabulary = SLCVOrbVocabulary::get();

    //instantiate and load slam map
    mpKeyFrameDatabase = new SLCVKeyFrameDB(*mpVocabulary);

    _map = new SLCVMap("Map");

    //setup file system and check for existing files
    SLCVMapStorage::init();
    //make new map
    SLCVMapStorage::newMap();

    //set SLCVMap in SLCVMapNode and update SLCVMapNode with scene objects
    mapNode->setMap(*_map);
    //the map is rotated w.r.t world because ORB-SLAM uses x-axis right, 
    //y-axis down and z-forward
    mapNode->rotate(180, 1, 0, 0);
    //the tracking camera has to be a child of the slam map, 
    //because we manipulate its position (object matrix) in the maps coordinate system
    mapNode->addChild(node);

    if (_map->KeyFramesInMap())
        _initialized = true;
    else
        _initialized = false;

    int nFeatures = 1000;
    float fScaleFactor = 1.2;
    int nLevels = 8;
    int fIniThFAST = 20;
    int fMinThFAST = 7;

    //instantiate Orb extractor
    _extractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    //instantiate local mapping
    mpLocalMapper = new LocalMapping(_map, 1, mpVocabulary, mbOnlyTracking);
    mpLoopCloser = new LoopClosing(_map, mpKeyFrameDatabase, mpVocabulary, false);

    mpLocalMapper->SetLoopCloser(mpLoopCloser);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    if (!_serial)
    {
        mptLocalMapping = new thread(&LocalMapping::Run, mpLocalMapper);
        mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);
    }
}
//-----------------------------------------------------------------------------
SLCVTrackedMapping::~SLCVTrackedMapping()
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
        delete _extractor;
    if (mpIniORBextractor)
        delete mpIniORBextractor;
    if (mpLocalMapper)
        delete mpLocalMapper;
    if (mpLoopCloser)
        delete mpLoopCloser;
}
//-----------------------------------------------------------------------------
SLbool SLCVTrackedMapping::track(SLCVMat imageGray,
                               SLCVMat imageRgb,
                               SLCVCalibration* calib,
                               SLbool drawDetection,
                               SLSceneView* sv)
{
    //store reference to current color image for decoration
    _img = imageRgb;

    // Current Frame
    double timestamp = 0.0; //todo

    _calib = calib;
    _imageGray = imageGray;
    SLCVMapTracking::track();

    return false;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::initialize()
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

    mCurrentFrame = SLCVFrame(_imageGray, 0.0, mpIniORBextractor,
        _calib->cameraMat(), _calib->distortion(), mpVocabulary, _retainImg);

    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = SLCVFrame(mCurrentFrame);
            mLastFrame = SLCVFrame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            //ghm1: we store the undistorted keypoints of the initial frame in an extra vector 
            //todo: why not using mInitialFrame.mvKeysUn????
            for (size_t i = 0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            // TODO(jan): is this necessary?
            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new ORB_SLAM2::Initializer(mCurrentFrame, 1.0, 200);
            //ghm1: clear mvIniMatches. it contains the index of the matched keypoint in the current frame
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            initializationStatus = INITIALIZATION_STATUS_REFERENCE_KEYFRAME_SET;

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
            initializationStatus = INITIALIZATION_STATUS_REFERENCE_KEYFRAME_DELETED_KEYPOINTS;
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            initializationStatus = INITIALIZATION_STATUS_REFERENCE_KEYFRAME_DELETED_MATCHES;
            return;
        }

        //ghm1: decorate image with tracked matches
        for (unsigned int i = 0; i<mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] >= 0)
            {
                cv::line(_img, mInitialFrame.mvKeys[i].pt, mCurrentFrame.mvKeys[mvIniMatches[i]].pt,
                    cv::Scalar(0, 255, 0));
            }
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            initializationStatus = INITIALIZATION_STATUS_INITIALIZER_INITIALIZED;

            for (size_t i = 0, iend = mvIniMatches.size(); i<iend; i++)
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

            bool mapInitializedSuccessfully = CreateInitialMapMonocular();
            if (mapInitializedSuccessfully)
            {
                //mark tracking as initialized
                _initialized = true;
                _bOK = true;
                initializationStatus = INITIALIZATION_STATUS_INITIALIZED;
            }

            //ghm1: in the original implementation the initialization is defined in the track() function and this part is always called at the end!
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if (!mCurrentFrame.mTcw.empty() && mCurrentFrame.mpReferenceKF)
            {
                cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr);
                mlpReferences.push_back(mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(sm.state() == SLCVTrackingStateMachine::TRACKING_LOST);
            }
            else if(mlRelativeFramePoses.size())
            {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(sm.state() == SLCVTrackingStateMachine::TRACKING_LOST);
            }
        }
    }


}
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::TrackWithOptFlow()
{
    SLAverageTiming::start("TrackWithOptFlow");

    if (mLastFrame.mvKeys.size() < 100)
    {
        SLAverageTiming::stop("TrackWithOptFlow");
        return false;
    }

    SLCVMat rvec = SLCVMat::zeros(3, 1, CV_64FC1);
    SLCVMat tvec = SLCVMat::zeros(3, 1, CV_64FC1);
    cv::Mat om = mLastFrame.mTcw;
    Rodrigues(om.rowRange(0, 3).colRange(0, 3), rvec);
    tvec = om.colRange(3, 4).rowRange(0, 3);

    SLVuchar status;
    SLVfloat err;
    SLCVSize winSize(15, 15);

    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                              1,    // terminate after this many iterations, or
                              0.03); // when the search window moves by less than this

    SLCVVPoint2f keyPointCoordinatesLastFrame;
    vector<SLCVMapPoint*> matchedMapPoints;
    vector<cv::KeyPoint> matchedKeyPoints;

    if(_optFlowOK)
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

    // Find closest possible feature points based on optical flow
    SLCVVPoint2f pred2DPoints(keyPointCoordinatesLastFrame.size());

    cv::calcOpticalFlowPyrLK(
        mLastFrame.imgGray,             // Previous frame
        mCurrentFrame.imgGray,          // Current frame
        keyPointCoordinatesLastFrame,   // Previous and current keypoints coordinates.The latter will be
        pred2DPoints,                   // expanded if more good coordinates are detected during OptFlow
        status,                         // Output vector for keypoint correspondences (1 = match found)
        err,                            // Error size for each flow
        winSize,                        // Search window for each pyramid level
        3,                              // Max levels of pyramid creation
        criteria,                       // Configuration from above
        0,                              // Additional flags
        0.01);                         // Minimal Eigen threshold

    // Only use points which are not wrong in any way during the optical flow calculation
    SLCVVPoint2f frame2DPoints;
    SLCVVPoint3f model3DPoints;

    vector<SLCVMapPoint*> trackedMapPoints;
    vector<cv::KeyPoint> trackedKeyPoints;

    mnMatchesInliers = 0;

    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            // TODO(jan): if pred2DPoints is really expanded during optflow, then the association
            // to 3D points is maybe broken?
            frame2DPoints.push_back(pred2DPoints[i]);
            SLVec3f v = matchedMapPoints[i]->worldPosVec();
            cv::Point3f p3(v.x, v.y, v.z);
            model3DPoints.push_back(p3);

            matchedKeyPoints[i].pt.x = pred2DPoints[i].x;
            matchedKeyPoints[i].pt.y = pred2DPoints[i].y;

            trackedMapPoints.push_back(matchedMapPoints[i]);
            trackedKeyPoints.push_back(matchedKeyPoints[i]);

            std::lock_guard<std::mutex> guard(_nMapMatchesLock);
            mnMatchesInliers++;
        }
    }

    if (trackedKeyPoints.size() < matchedKeyPoints.size() * 0.75)
    {
        SLAverageTiming::stop("TrackWithOptFlow");
        return false;
    }

    /////////////////////
    // Pose Estimation //
    /////////////////////

    bool foundPose = cv::solvePnP(model3DPoints,
                                  frame2DPoints,
                                  _calib->cameraMat(),
                                  _calib->distortion(),
                                  rvec, tvec,
                                  true);

    if (foundPose)
    {
        SLCVMat Tcw = cv::Mat::eye(4, 4, CV_32F);
        Tcw.at<float>(0, 3) = tvec.at<float>(0, 0);
        Tcw.at<float>(1, 3) = tvec.at<float>(1, 0);
        Tcw.at<float>(2, 3) = tvec.at<float>(2, 0);

        SLCVMat Rcw = cv::Mat::zeros(3, 3, CV_32F);
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
        _optFlowTcw = Tcw;

        _optFlowMapPtsLastFrame = trackedMapPoints;
        _optFlowKeyPtsLastFrame = trackedKeyPoints;

        trackingType = TrackingType_OptFlow;
    }

    SLAverageTiming::stop("TrackWithOptFlow");

    return foundPose;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::track3DPts()
{
    mCurrentFrame = SLCVFrame(_imageGray, 0.0, _extractor,
        _calib->cameraMat(), _calib->distortion(), mpVocabulary, _retainImg);

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(_map->mMutexMapUpdate, std::defer_lock);
    if (!_serial)
    {
        lock.lock();
    }

    //mLastProcessedState = mState;
    //bool bOK;
    _bOK = false;
    trackingType = TrackingType_None;

    // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
    if(!mbOnlyTracking)
    {
        // Local Mapping is activated. This is the normal behaviour, unless
        // you explicitly activate the "only tracking" mode.

        if(sm.state() == SLCVTrackingStateMachine::TRACKING_OK)
        {
            // Local Mapping might have changed some MapPoints tracked in last frame
            CheckReplacedInLastFrame();

            if(mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                _bOK = TrackReferenceKeyFrame();
                trackingType = TrackingType_ORBSLAM;
            }
            else
            {
                _bOK = TrackWithMotionModel();
                trackingType = TrackingType_MotionModel;

                if(!_bOK)
                {
                    _bOK = TrackReferenceKeyFrame();
                    trackingType = TrackingType_ORBSLAM;
                }
            }
        }
        else
        {
            _bOK = Relocalization();
        }
    }
    else
    {
        // Localization Mode: Local Mapping is deactivated

        if (sm.state() == SLCVTrackingStateMachine::TRACKING_LOST)
        {
            _bOK = Relocalization();
            _optFlowOK = false;
            //cout << "Relocalization: " << bOK << endl;
        }
        else
        {
            if (sm.state() == SLCVTrackingStateMachine::TRACKING_OK)
            {
                //We always run the optical flow additionally, because it gives
                //a more stable pose. We use this pose if successful.
                _optFlowOK = TrackWithOptFlow();
            }

            //if NOT visual odometry tracking
            if (!mbVO) // In last frame we tracked enough MapPoints from the Map
            {
                if (!mVelocity.empty())
                { //we have a valid motion model
                    _bOK = TrackWithMotionModel();
                    if(!_optFlowOK)
                        trackingType = TrackingType_MotionModel;
                    //cout << "TrackWithMotionModel: " << bOK << endl;
                }
                else
                {
                    //we have NO valid motion model
                        // All keyframes that observe a map point are included in the local map.
                        // Every current frame gets a reference keyframe assigned which is the keyframe
                        // from the local map that shares most matches with the current frames local map points matches.
                        // It is updated in UpdateLocalKeyFrames().
                    _bOK = TrackReferenceKeyFrame();
                    if (!_optFlowOK)
                        trackingType = TrackingType_ORBSLAM;
                    //cout << "TrackReferenceKeyFrame" << endl;
                }
            }
            else // In last frame we tracked mainly "visual odometry" points.
            {
                // We compute two camera poses, one from motion model and one doing relocalization.
                // If relocalization is sucessfull we choose that solution, otherwise we retain
                // the "visual odometry" solution.
                bool bOKMM = false;
                bool bOKReloc = false;
                vector<SLCVMapPoint*> vpMPsMM;
                vector<bool> vbOutMM;
                cv::Mat TcwMM;
                if (!mVelocity.empty())
                {
                    bOKMM = TrackWithMotionModel();
                    vpMPsMM = mCurrentFrame.mvpMapPoints;
                    vbOutMM = mCurrentFrame.mvbOutlier;
                    TcwMM = mCurrentFrame.mTcw.clone();
                }
                bOKReloc = Relocalization();

                //relocalization method is not valid but the velocity model method
                if (bOKMM && !bOKReloc)
                {
                    mCurrentFrame.SetPose(TcwMM);
                    mCurrentFrame.mvpMapPoints = vpMPsMM;
                    mCurrentFrame.mvbOutlier = vbOutMM;

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
                trackingType = TrackingType_None;
            }
        }
    }

    // If we have an initial estimation of the camera pose and matching. Track the local map.
    if(!mbOnlyTracking)
    {
        if(_bOK)
        {
            _bOK = TrackLocalMap();
        }
    }
    else
    {
        // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
        // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
        // the camera we will use the local map again.
        if (_bOK && !mbVO)
            _bOK = TrackLocalMap();
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
            mVelocity = mCurrentFrame.mTcw*LastTwc;
        }
        else
        {
            mVelocity = cv::Mat();
        }

        //set current pose
        {
            cv::Mat Tcw;
            if(_optFlowOK)
            {
                Tcw = _optFlowTcw.clone();
            }
            else
            {
                Tcw = mCurrentFrame.mTcw.clone();
            }

            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            //inversion
            //auto Tcw = mCurrentFrame.mTcw.clone();
            Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc*Tcw.rowRange(0, 3).col(3);

            //conversion to SLMat4f
            SLMat4f slMat((SLfloat)Rwc.at<float>(0, 0), (SLfloat)Rwc.at<float>(0, 1), (SLfloat)Rwc.at<float>(0, 2), (SLfloat)twc.at<float>(0, 0),
                (SLfloat)Rwc.at<float>(1, 0), (SLfloat)Rwc.at<float>(1, 1), (SLfloat)Rwc.at<float>(1, 2), (SLfloat)twc.at<float>(1, 0),
                (SLfloat)Rwc.at<float>(2, 0), (SLfloat)Rwc.at<float>(2, 1), (SLfloat)Rwc.at<float>(2, 2), (SLfloat)twc.at<float>(2, 0),
                0.0f, 0.0f, 0.0f, 1.0f);
            slMat.rotate(180, 1, 0, 0);
            // set the object matrix of this object (its a SLCamera)
            _node->om(slMat);
        }

        // Clean VO matches
        for (int i = 0; i<mCurrentFrame.N; i++)
        {
            SLCVMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (pMP->Observations()<1)
                {
                    mCurrentFrame.mvbOutlier[i] = false;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
                }
            }
        }

        //ghm1: manual local mapping of current frame
        if ((_bOK && _mapNextFrame) || NeedNewKeyFrame())
        {
            CreateNewKeyFrame();

            if (_serial)
            {
                //call local mapper
                mpLocalMapper->RunOnce();
                //normally the loop closing would feed the keyframe database, but we do it here
                //mpKeyFrameDatabase->add(mpLastKeyFrame);

                //loop closing
                mpLoopCloser->RunOnce();
            }

            _mapNextFrame = false;
            //update visualization of map, it may have changed because of global bundle adjustment.
            //map points will be updated with next decoration.
            _mapHasChanged = true;
        }

        // We allow points with high innovation (considererd outliers by the Huber Function)
        // pass to the new keyframe, so that bundle adjustment will finally decide
        // if they are outliers or not. We don't want next frame to estimate its position
        // with those points so we discard them in the frame.
        for (int i = 0; i<mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
            }
        }
    }

    if (!mCurrentFrame.mpReferenceKF)
    {
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    decorate();

    mLastFrame = SLCVFrame(mCurrentFrame);

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (mCurrentFrame.mpReferenceKF && !mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse(); //Tcr = Tcw * Twr (current wrt reference = world wrt current * reference wrt world
                                                                                        //relative frame poses are used to refer a frame to reference frame
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(sm.state() == SLCVTrackingStateMachine::TRACKING_LOST);
    }
    else if (mlRelativeFramePoses.size() && mlpReferences.size() && mlFrameTimes.size())
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(sm.state() == SLCVTrackingStateMachine::TRACKING_LOST);
    }
}
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::CreateInitialMapMonocular()
{
    // Create KeyFrames
    SLCVKeyFrame* pKFini = new SLCVKeyFrame(mInitialFrame, _map, mpKeyFrameDatabase);
    SLCVKeyFrame* pKFcur = new SLCVKeyFrame(mCurrentFrame, _map, mpKeyFrameDatabase);

    cout << "pKFini num keypoints: " << mInitialFrame.N << endl;
    cout << "pKFcur num keypoints: " << mCurrentFrame.N << endl;

    pKFini->ComputeBoW( mpVocabulary );
    pKFcur->ComputeBoW( mpVocabulary );

    // Insert KFs in the map
    _map->AddKeyFrame(pKFini);
    _map->AddKeyFrame(pKFcur);
    //mpKeyFrameDatabase->add(pKFini);
    //mpKeyFrameDatabase->add(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i<mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        SLCVMapPoint* pMP = new SLCVMapPoint(worldPos, pKFcur, _map);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

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
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return false;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
    //pKFcur->SetPose(Tc2w);
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<SLCVMapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            SLCVMapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = _map->GetAllMapPoints();

    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = SLCVFrame(mCurrentFrame);

    _map->SetReferenceMapPoints(mvpLocalMapPoints);

    _map->mvpKeyFrameOrigins.push_back(pKFini);

    //mState = OK;
    /*_bOK = true;*/
    //_currentState = TRACK_3DPTS;

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
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        SLCVMapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            SLCVMapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::NeedNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = _map->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Thresholds
    float thRefRatio = 0.9f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio) && mnMatchesInliers>15);

    if((c1a||c1b)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
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
        return false;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    SLCVKeyFrame* pKF = new SLCVKeyFrame(mCurrentFrame, _map, mpKeyFrameDatabase);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    //if (mSensor != System::MONOCULAR)
    //{
    //    mCurrentFrame.UpdatePoseMatrices();

    //    // We sort points by the measured depth by the stereo/RGBD sensor.
    //    // We create all those MapPoints whose depth < mThDepth.
    //    // If there are less than 100 close points we create the 100 closest.
    //    vector<pair<float, int> > vDepthIdx;
    //    vDepthIdx.reserve(mCurrentFrame.N);
    //    for (int i = 0; i<mCurrentFrame.N; i++)
    //    {
    //        float z = mCurrentFrame.mvDepth[i];
    //        if (z>0)
    //        {
    //            vDepthIdx.push_back(make_pair(z, i));
    //        }
    //    }

    //    if (!vDepthIdx.empty())
    //    {
    //        sort(vDepthIdx.begin(), vDepthIdx.end());

    //        int nPoints = 0;
    //        for (size_t j = 0; j<vDepthIdx.size(); j++)
    //        {
    //            int i = vDepthIdx[j].second;

    //            bool bCreateNew = false;

    //            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
    //            if (!pMP)
    //                bCreateNew = true;
    //            else if (pMP->Observations()<1)
    //            {
    //                bCreateNew = true;
    //                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    //            }

    //            if (bCreateNew)
    //            {
    //                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
    //                MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
    //                pNewMP->AddObservation(pKF, i);
    //                pKF->AddMapPoint(pNewMP, i);
    //                pNewMP->ComputeDistinctiveDescriptors();
    //                pNewMP->UpdateNormalAndDepth();
    //                mpMap->AddMapPoint(pNewMP);

    //                mCurrentFrame.mvpMapPoints[i] = pNewMP;
    //                nPoints++;
    //            }
    //            else
    //            {
    //                nPoints++;
    //            }

    //            if (vDepthIdx[j].first>mThDepth && nPoints>100)
    //                break;
    //        }
    //    }
    //}

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::Reset()
{

    cout << "System Reseting" << endl;
//    if (mpViewer)
//    {
//        mpViewer->RequestStop();
//        while (!mpViewer->isStopped()) {
//#ifdef WINDOWS
//            Sleep(3);
//#else
//            usleep(3000);
//#endif
//        }
//    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    if (!_serial)
    {
        mpLocalMapper->RequestReset();
    }
    else
    {
        mpLocalMapper->reset();
    }
    cout << " done" << endl;

    //// Reset Loop Closing
    //cout << "Reseting Loop Closing...";
    if (!_serial)
    {
        mpLoopCloser->RequestReset();
    }
    else
    {
        mpLoopCloser->reset();
    }
    //cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    //mpKeyFrameDB->clear();
    mpKeyFrameDatabase->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    //mpMap->clear();
    _map->clear();

    SLCVKeyFrame::nNextId = 0;
    SLCVFrame::nNextId = 0;
    //mState = NO_IMAGES_YET;
    _bOK = false;
    _initialized = false;

    if (mpInitializer)
    {
        cout << "Resetting Initializer...";
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
        cout << " done" << endl;
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    mpLastKeyFrame = NULL;
    mpReferenceKF = NULL;
    mvpLocalMapPoints.clear();
    mvpLocalKeyFrames.clear();
    mnMatchesInliers = 0;
    _addKeyframe = false;

    //we also have to clear the mapNode because it may access
    //mappoints and keyframes while we are loading
    _mapNode->clearAll();
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::Pause()
{
    if (!_serial)
    {
        mpLocalMapper->RequestStop();
        while (!mpLocalMapper->isStopped())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    sm.requestStateIdle();
    while (!sm.hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::Resume()
{
    if (!_serial)
    {
        mpLocalMapper->Release();
        //mptLocalMapping = new thread(&LocalMapping::Run, mpLocalMapper);
        //mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);
    }

    sm.requestResume();
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::decorate()
{
    //calculation of mean reprojection error of all matches
    calculateMeanReprojectionError();
    //calculate pose difference
    calculatePoseDifference();
    //show rectangle for key points in video that where matched to map points
    decorateVideoWithKeyPointMatches(_img);
    //decorate scene with matched map points, local map points and matched map points
    decorateScene();
}
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query SLCVKeyFrame Database for keyframe candidates for relocalisation
    vector<SLCVKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    //vector<SLCVKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->keyFrames();
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver

    ORBmatcher matcher(0.75, true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<SLCVMapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i<nKFs; i++)
    {
        SLCVKeyFrame* pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            //cout << "Num matches: " << nmatches << endl;
            if (nmatches<15)
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
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates>0 && !bMatch)
    {
        for (int i = 0; i<nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

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

                set<SLCVMapPoint*> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j<np; j++)
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

                if (nGood<10)
                    continue;

                for (int io = 0; io<mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<SLCVMapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again:
                //ghm1: mappoints seen in the keyframe which was found as candidate via BoW-search are projected into
                //the current frame using the position that was calculated using the matches from BoW matcher
                if (nGood<50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip<mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io<mCurrentFrame.N; io++)
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
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::TrackReferenceKeyFrame()
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
    ORBmatcher matcher(0.7, true);
    vector<SLCVMapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i<mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                SLCVMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap >= 10;
}
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::TrackLocalMap()
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
    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i<mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations()>0)
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
            //    mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50) {
        //cout << "mnMatchesInliers: " << mnMatchesInliers << endl;
        return false;
    }

    if (mnMatchesInliers<30)
        return false;
    else
        return true;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::UpdateLocalMap()
{
    // This is for visualization
    //mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<SLCVMapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        SLCVMapPoint* pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<SLCVMapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<SLCVMapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        SLCVMapPoint* pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills SLCVMapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            //ghm1 test:
            //if (!_image.empty())
            //{
            //    SLCVPoint2f ptProj(pMP->mTrackProjX, pMP->mTrackProjY);
            //    cv::rectangle(_image,
            //        cv::Rect(ptProj.x - 3, ptProj.y - 3, 7, 7),
            //        Scalar(0, 0, 255));
            //}
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId<mnLastRelocFrameId + 2)
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<SLCVKeyFrame*, int> keyframeCounter;
    for (int i = 0; i<mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            SLCVMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const map<SLCVKeyFrame*, size_t> observations = pMP->GetObservations();
                for (map<SLCVKeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
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

    int max = 0;
    SLCVKeyFrame* pKFmax = static_cast<SLCVKeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<SLCVKeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        SLCVKeyFrame* pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second>max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<SLCVKeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size()>80)
            break;

        SLCVKeyFrame* pKF = *itKF;

        const vector<SLCVKeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<SLCVKeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            SLCVKeyFrame* pNeighKF = *itNeighKF;
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

        const set<SLCVKeyFrame*> spChilds = pKF->GetChilds();
        for (set<SLCVKeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            SLCVKeyFrame* pChildKF = *sit;
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

        SLCVKeyFrame* pParent = pKF->GetParent();
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
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (vector<SLCVKeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        SLCVKeyFrame* pKF = *itKF;
        const vector<SLCVMapPoint*> vpMPs = pKF->GetMapPointMatches();

        for (vector<SLCVMapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            SLCVMapPoint* pMP = *itMP;
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
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::TrackWithMotionModel()
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
    UpdateLastFrame();

    //this adds the motion differnce between the last and the before-last frame to the pose of the last frame to estimate the position of the current frame
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<SLCVMapPoint*>(NULL));

    // Project points seen in previous frame
    int th = 15;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, true);

    // If few matches, uses a wider window search
    if (nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<SLCVMapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, true);
    }

    if (nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i<mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                SLCVMapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap >= 10;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    SLCVKeyFrame* pRef = mLastFrame.mpReferenceKF;
    //cout << "pRef pose: " << pRef->GetPose() << endl;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    //GHM1:
    //l = last, w = world, r = reference
    //Tlr is the relative transformation for the last frame wrt to reference frame
    //(because the relative pose for the current frame is added at the end of tracking)
    //Refer last frame pose to world: Tlw = Tlr * Trw
    //So it seems, that the frames pose does not always refer to world frame...?
    mLastFrame.SetPose(Tlr*pRef->GetPose());
}
////-----------------------------------------------------------------------------
//void SLCVTrackedMapping::saveMap()
//{
//    SLCVMapStorage::saveMap(SLCVMapStorage ::getCurrentId(),*_map, true);
//    //update key frames, because there may be new textures in file system
//    _mapNode->updateKeyFrames(_map->GetAllKeyFrames());
//}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::globalBundleAdjustment()
{
    Optimizer::GlobalBundleAdjustemnt(_map, 20);
    _mapNode->updateAll(*_map);
}
//-----------------------------------------------------------------------------
size_t SLCVTrackedMapping::getSizeOf()
{
    size_t size = 0;

    size += sizeof(*this);
    //add size of local mapping
    //add size of loop closing
    //add size of 

    return size;
}

SLCVKeyFrame* SLCVTrackedMapping::currentKeyFrame()
{
    SLCVKeyFrame* result = mCurrentFrame.mpReferenceKF;

    return result;
}

const char* SLCVTrackedMapping::getLoopClosingStatusString()
{
    switch (mpLoopCloser->status())
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
        case LoopClosing::LOOP_CLOSE_STATUS_NO_OPTIMIZED_CANDIDATES:
            return "no optimized candidates";
        case LoopClosing::LOOP_CLOSE_STATUS_NO_NEW_KEYFRAME:
            return "no new keyframe";
        case LoopClosing::LOOP_CLOSE_STATUS_NONE:
        default:
            return "none";
    }
}

const char* SLCVTrackedMapping::getInitializationStatusString()
{
    switch (initializationStatus)
    {
        case INITIALIZATION_STATUS_INITIALIZED:
            return "initialized";
        case INITIALIZATION_STATUS_REFERENCE_KEYFRAME_SET:
            return "reference keyframe set";
        case INITIALIZATION_STATUS_REFERENCE_KEYFRAME_DELETED_KEYPOINTS:
            return "reference keyframe deleted - not enough keypoints";
        case INITIALIZATION_STATUS_REFERENCE_KEYFRAME_DELETED_MATCHES:
            return "reference keyframe deleted - not enough matches";
        case INITIALIZATION_STATUS_INITIALIZER_INITIALIZED:
            return "initializer initialized";
        case INITIALIZATION_STATUS_NONE:
        default:
            return "none";
    }
}
