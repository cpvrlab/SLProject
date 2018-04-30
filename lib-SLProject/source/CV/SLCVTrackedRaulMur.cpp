//#############################################################################
//  File:      SLCVTrackedRaulMur.cpp
//  Author:    Michael G�ttlicher
//  Date:      Dez 2017
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
#include <SLCVTrackedRaulMur.h>
#include <SLCVFrame.h>
#include <SLPoints.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/PnPsolver.h>
#include <OrbSlam/Optimizer.h>
#include <SLAverageTiming.h>
#include <SLCVMapNode.h>

using namespace cv;
using namespace ORB_SLAM2;

//-----------------------------------------------------------------------------
SLCVTrackedRaulMur::SLCVTrackedRaulMur(SLNode *node, ORBVocabulary* vocabulary, 
    SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLCVMapNode* mapNode)
    : SLCVTracked(node),
    SLCVMapTracking(keyFrameDB, map, mapNode, true),
    mpVocabulary(vocabulary)
    //mpKeyFrameDatabase(keyFrameDB),
    //_map(map),
    //_mapNode(mapNode)
{
    //instantiate Orb extractor
    _extractor = new ORBextractor(1500, 1.44f, 4, 30, 20);

    //system is initialized, because we loaded an existing map, but we have to relocalize
    //mState = LOST;
    _initialized = true;
    _bOK = false;
}
//-----------------------------------------------------------------------------
SLCVTrackedRaulMur::~SLCVTrackedRaulMur()
{
    if (_extractor)
        delete _extractor;
}
//-----------------------------------------------------------------------------
SLbool SLCVTrackedRaulMur::track(SLCVMat imageGray,
    SLCVMat imageRgb,
    SLCVCalibration* calib,
    SLbool drawDetection,
    SLSceneView* sv)
{



    SLAverageTiming::start("track");

    //SLCVMat imageGrayResized;
    //cv::resize(imageGray, imageGrayResized,
    //    cv::Size(640, 480));

    //_image = image;
    if (_frameCount == 0) {
        _calib = calib;
    }
    _frameCount++;

    //SLCVMat imageGrayScaled;
    //imageGray.copyTo(imageGrayScaled);
    //cv::resize(imageGrayScaled, imageGrayScaled, 640.0, 480.0);

    SLAverageTiming::start("newFrame");
    /************************************************************/
    //Frame constructor call in ORB-SLAM:
    // Current Frame
    double timestamp = 0.0; //todo
    mCurrentFrame = SLCVFrame(imageGray, timestamp, _extractor,
        calib->cameraMat(), calib->distortion(), mpVocabulary );
    /************************************************************/
    SLAverageTiming::stop("newFrame");

    //undistort color video image
    //calib->remap(image, image);

    // System is initialized. Track Frame.
    //mLastProcessedState = mState;
    //bool bOK=false;

    // Localization Mode: Local Mapping is deactivated
    if (sm.state() == SLCVTrackingStateMachine::TRACKING_LOST)
    {
        SLAverageTiming::start("Relocalization");
        _bOK = Relocalization();
        SLAverageTiming::stop("Relocalization");
    }
    else
    {
        //if NOT visual odometry tracking
        if (!mbVO) // In last frame we tracked enough MapPoints from the Map
        {
            if (!mVelocity.empty()) { //we have a valid motion model
                SLAverageTiming::start("TrackWithMotionModel");
                _bOK = TrackWithMotionModel();
                SLAverageTiming::stop("TrackWithMotionModel");
            }
            else { //we have NO valid motion model
                // All keyframes that observe a map point are included in the local map.
                // Every current frame gets a reference keyframe assigned which is the keyframe 
                // from the local map that shares most matches with the current frames local map points matches.
                // It is updated in UpdateLocalKeyFrames().
                SLAverageTiming::start("TrackReferenceKeyFrame");
                _bOK = TrackReferenceKeyFrame();
                SLAverageTiming::stop("TrackReferenceKeyFrame");
            }
        }
        else // In last frame we tracked mainly "visual odometry" points.
        {
            // We compute two camera poses, one from motion model and one doing relocalization.
            // If relocalization is sucessfull we choose that solution, otherwise we retain
            // the "visual odometry" solution.
            SLAverageTiming::start("visualOdometry");

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
                    for (int i = 0; i<mCurrentFrame.N; i++)
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

            SLAverageTiming::stop("visualOdometry");
        }
    }


    // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
    // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
    // the camera we will use the local map again.

    if (_bOK && !mbVO)
    {
        SLAverageTiming::start("TrackLocalMap");
        _bOK = TrackLocalMap();
        SLAverageTiming::stop("TrackLocalMap");
    }

    //if (bOK)
    //    mState = OK;
    //else
    //    mState = LOST;

    //add map points to scene and keypoints to video image
    decorateSceneAndVideo(imageRgb);

    // If tracking were good
    if (_bOK)
    {
        // Update motion model
        if (!mLastFrame.mTcw.empty())
        {
            cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
            mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3)); //mRwc
            const auto& cc = mLastFrame.GetCameraCenter(); //this is the translation w.r.t the world of the frame (warum dann Twc??)
            cc.copyTo(LastTwc.rowRange(0, 3).col(3));
            mVelocity = mCurrentFrame.mTcw*LastTwc;
        }
        else
            mVelocity = cv::Mat();

        //set current pose
        {
            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            //inversion
            auto Tcw = mCurrentFrame.mTcw.clone();
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
                if (pMP->Observations()<1)
                {
                    mCurrentFrame.mvbOutlier[i] = false;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
                }
        }

        // We allow points with high innovation (considererd outliers by the Huber Function)
        // pass to the new keyframe, so that bundle adjustment will finally decide
        // if they are outliers or not. We don't want next frame to estimate its position
        // with those points so we discard them in the frame.
        for (int i = 0; i<mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
        }
    }


    if (!mCurrentFrame.mpReferenceKF)
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

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
    else if(mlRelativeFramePoses.size() && mlpReferences.size() && mlFrameTimes.size())
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(sm.state() == SLCVTrackingStateMachine::TRACKING_LOST);
    }
    
    SLAverageTiming::stop("track");
    return false;
}
//-----------------------------------------------------------------------------
bool SLCVTrackedRaulMur::Relocalization()
{
    //ghm1:
    //The goal is to find a camera pose of the current frame with more than 50 matches of keypoints to mappoints
    //1. search for relocalization candidates by querying the keyframe database (with similarity score)
    //2. for every found candidate we search keypoint matches (from kf candidate) with orb in current frame (ORB that belong to the same vocabulary node (at a certain level))
    //3. if more than 15 matches are found we use a PnPSolver with RANSAC to estimate an initial camera pose
    //4. if the pose is valid (RANSAC has not reached max. iterations), pose and matches are inserted into current frame and the pose of the frame is optimized using optimizer
    //5. if more less than 10 good matches remain continue with next candidate
    //6. else if less than 50 good matched remained after optimization, mappoints associated with the keyframe candidate are projected in the current frame (which has an initial pose) and more matches are searched in a coarse window
    //7. if we now have found more than 50 matches the pose of the current frame is optimized again using the additional found matches
    //8. during the optimization matches may be rejected. so if after optimization more than 30 and less than 50 matches remain we search again by projection using a narrower search window
    //9. if now more than 50 matches exist after search by projection the pose is optimized again (for the last time)
    //10. if more than 50 good matches remain after optimization, relocalization was successful

    // Compute Bag of Words Vector
    SLAverageTiming::start("ComputeBoW");
    mCurrentFrame.ComputeBoW();
    SLAverageTiming::stop("ComputeBoW");

    // Relocalization is performed when tracking is lost
    // Track Lost: Query SLCVKeyFrame Database for keyframe candidates for relocalisation
    SLAverageTiming::start("DetectRelocalizationCandidates");
    vector<SLCVKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);
    SLAverageTiming::stop("DetectRelocalizationCandidates");

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver

    SLAverageTiming::start("MatchCandsAndSolvePose");
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
    SLAverageTiming::stop("MatchCandsAndSolvePose");

    SLAverageTiming::start("SearchCandsUntil50Matches");
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
    SLAverageTiming::stop("SearchCandsUntil50Matches");

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
bool SLCVTrackedRaulMur::TrackWithMotionModel()
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

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<SLCVMapPoint*>(NULL));

    // Project points seen in previous frame
    SLAverageTiming::start("SearchByProjection7");
    int th = 15;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, true);
    SLAverageTiming::stop("SearchByProjection7");

    // If few matches, uses a wider window search
    SLAverageTiming::start("SearchByProjection14");
    if (nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<SLCVMapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, true);
    }
    SLAverageTiming::stop("SearchByProjection14");

    if (nmatches<20)
        return false;

    SLAverageTiming::start("PoseOptimizationTWMM");
    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);
    SLAverageTiming::stop("PoseOptimizationTWMM");

    SLAverageTiming::start("DiscardOutliers");
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
    SLAverageTiming::stop("DiscardOutliers");

    //if (mbOnlyTracking)
    //{
    mbVO = nmatchesMap<10;
    return nmatches>20;
    //}


    //return nmatchesMap >= 10;
}
//-----------------------------------------------------------------------------
bool SLCVTrackedRaulMur::TrackLocalMap()
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
    SLAverageTiming::start("UpdateLocalMap");
    UpdateLocalMap();
    SLAverageTiming::stop("UpdateLocalMap");

    SLAverageTiming::start("SearchLocalPoints");
    SearchLocalPoints();
    SLAverageTiming::stop("SearchLocalPoints");

    // Optimize Pose
    SLAverageTiming::start("PoseOptimizationTLM");
    Optimizer::PoseOptimization(&mCurrentFrame);
    SLAverageTiming::stop("PoseOptimizationTLM");
    mnMatchesInliers = 0;

    //SLAverageTiming::start("UpdateMapPointsStat");
    // Update MapPoints Statistics
    for (int i = 0; i<mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    mnMatchesInliers++;
            }
            //else if (mSensor == System::STEREO)
            //    mCurrentFrame.mvpMapPoints[i] = static_cast<SLCVMapPoint*>(NULL);
        }
    }
    //SLAverageTiming::stop("UpdateMapPointsStat");

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId<mnLastRelocFrameId + mMaxFrames && mnMatchesInliers<50)
        return false;

    if (mnMatchesInliers<30)
        return false;
    else
        return true;
}
//-----------------------------------------------------------------------------
void SLCVTrackedRaulMur::SearchLocalPoints()
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
bool SLCVTrackedRaulMur::TrackReferenceKeyFrame()
{
    //This routine is called if current tracking state is OK but we have NO valid motion model
    //1. Berechnung des BoW-Vectors fuer den current frame
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
void SLCVTrackedRaulMur::UpdateLastFrame()
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
//-----------------------------------------------------------------------------
void SLCVTrackedRaulMur::UpdateLocalMap()
{
    // This is for visualization
    //mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}
//-----------------------------------------------------------------------------
void SLCVTrackedRaulMur::UpdateLocalPoints()
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
void SLCVTrackedRaulMur::UpdateLocalKeyFrames()
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
void SLCVTrackedRaulMur::Reset()
{
    cout << "System Reseting" << endl;

    // Clear BoW Database
    mpKeyFrameDatabase->clear();

    // Clear Map (this erase MapPoints and KeyFrames)
    _map->clear();

    SLCVKeyFrame::nNextId = 0;
    SLCVFrame::nNextId = 0;
    //mState = NO_IMAGES_YET;
    _initialized = false;
    _bOK = false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    //mpLastKeyFrame = NULL;
    mpReferenceKF = NULL;
    mvpLocalMapPoints.clear();
    mvpLocalKeyFrames.clear();
    mnMatchesInliers = 0;
    //_addKeyframe = false;

    _mapNode->clearAll();
}