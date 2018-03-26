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
#include <OrbSlam/Initializer.h>
#include <SLCVKeyFrameDB.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
SLCVTrackedMapping::SLCVTrackedMapping(SLNode* node, ORBVocabulary* vocabulary, 
    SLCVKeyFrameDB* keyFrameDB, SLCVMap* map, SLNode* mapPC)
    : SLCVTracked(node),
    mpVocabulary(vocabulary),
    mpKeyFrameDatabase(keyFrameDB),
    _map(map),
    _mapPC(mapPC)
{
    //instantiate Orb extractor
    _extractor = new ORBextractor(1500, 1.44f, 4, 30, 20);
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
    if (_currentState != IDLE) {
        mCurrentFrame = SLCVFrame(imageGray, timestamp, _extractor,
            calib->cameraMat(), calib->distortion(), mpVocabulary);

        if ( true /*_showKeyPoints*/)
        {
            for (size_t i = 0; i < mCurrentFrame.N; i++)
            {
                //Use distorted points because we have to undistort the image later
                const auto& pt = mCurrentFrame.mvKeys[i].pt;
                cv::rectangle(_img,
                    cv::Rect(pt.x - 3, pt.y - 3, 7, 7),
                    cv::Scalar(0, 0, 255));
            }
        }
    }

    switch (_currentState)
    {
    case IDLE:
        break;
    case INITIALIZE:
        initialize();
        break;
    case TRACK_VO:
        trackVO();
        break;
    case TRACK_3DPTS:
        track3DPts();
        break;
    case TRACK_OPTICAL_FLOW:
        trackOpticalFlow();
        break;
    }

    return false;
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::initialize()
{
    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size()>100)
        {
            //ghm1
            Reset();

            mInitialFrame = SLCVFrame(mCurrentFrame);
            mLastFrame = SLCVFrame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new ORB_SLAM2::Initializer(mCurrentFrame, 1.0, 200);

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
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
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

            CreateInitialMapMonocular();
        }
    }
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::trackVO()
{

}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::track3DPts()
{
    mLastProcessedState = mState;
    bool bOK;

    if (mState == LOST)
    {
        bOK = Relocalization();
    }
    else
    {
        bOK = TrackReferenceKeyFrame();
    }

    // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
    // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
    // the camera we will use the local map again.

    SLAverageTiming::start("TrackLocalMap", 20, 1);
    if (bOK && !mbVO)
        bOK = TrackLocalMap();
    SLAverageTiming::stop("TrackLocalMap");

    if (bOK)
        mState = OK;
    else
        mState = LOST;

    // If tracking were good
    if (bOK)
    {
        //// Update motion model
        //if (!mLastFrame.mTcw.empty())
        //{
        //    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        //    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3)); //mRwc
        //    const auto& cc = mLastFrame.GetCameraCenter(); //this is the translation w.r.t the world of the frame (warum dann Twc??)
        //    cc.copyTo(LastTwc.rowRange(0, 3).col(3));
        //    mVelocity = mCurrentFrame.mTcw*LastTwc;
        //}
        //else
        //    mVelocity = cv::Mat();

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

    decorate();
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::trackOpticalFlow()
{

}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::CreateInitialMapMonocular()
{
    // Create KeyFrames
    SLCVKeyFrame* pKFini = new SLCVKeyFrame(mInitialFrame, _map, mpKeyFrameDatabase);
    SLCVKeyFrame* pKFcur = new SLCVKeyFrame(mCurrentFrame, _map, mpKeyFrameDatabase);

    pKFini->ComputeBoW( mpVocabulary );
    pKFcur->ComputeBoW( mpVocabulary );

    //// Insert KFs in the map
    //_map->AddKeyFrame(pKFini);
    //_map->AddKeyFrame(pKFcur);
    mpKeyFrameDatabase->add(pKFini);
    mpKeyFrameDatabase->add(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i<mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        SLCVMapPoint* pMP = new SLCVMapPoint(worldPos, pKFcur/*, _map*/);

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
        _map->mapPoints().push_back(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << _map->mapPoints().size() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(_map, 20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
    //pKFcur->SetPose(Tc2w);
    pKFcur->Tcw(Tc2w);

    // Scale points
    vector<SLCVMapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            SLCVMapPoint* pMP = vpAllMapPoints[iMP];
            //pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->worldPos(pMP->worldPos()*invMedianDepth);
        }
    }

    //mpLocalMapper->InsertKeyFrame(pKFini);
    //mpLocalMapper->InsertKeyFrame(pKFcur);

    //mCurrentFrame.SetPose(pKFcur->GetPose());
    //mnLastKeyFrameId = mCurrentFrame.mnId;
    //mpLastKeyFrame = pKFcur;

    //mvpLocalKeyFrames.push_back(pKFcur);
    //mvpLocalKeyFrames.push_back(pKFini);
    //mvpLocalMapPoints = mpMap->GetAllMapPoints();
    //mpReferenceKF = pKFcur;
    //mCurrentFrame.mpReferenceKF = pKFcur;

    //mLastFrame = Frame(mCurrentFrame);

    //mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    //mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    //mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    //mState = OK;

    _currentState = TRACK_3DPTS;
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

    //// Reset Local Mapping
    //cout << "Reseting Local Mapper...";
    //mpLocalMapper->RequestReset();
    //cout << " done" << endl;

    //// Reset Loop Closing
    //cout << "Reseting Loop Closing...";
    //mpLoopClosing->RequestReset();
    //cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    //mpKeyFrameDB->clear();
    mpKeyFrameDatabase->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    //mpMap->clear();
    _map->clear();

    //KeyFrame::nNextId = 0;
    //Frame::nNextId = 0;
    //mState = NO_IMAGES_YET;

    //if (mpInitializer)
    //{
    //    delete mpInitializer;
    //    mpInitializer = static_cast<Initializer*>(NULL);
    //}

    //mlRelativeFramePoses.clear();
    //mlpReferences.clear();
    //mlFrameTimes.clear();
    //mlbLost.clear();

    //if (mpViewer)
    //    mpViewer->Release();
}
//-----------------------------------------------------------------------------
void SLCVTrackedMapping::decorate()
{
    //draw map points
    if (_drawMapPoints && _mapPC) {
        //instantiate material
        if (!_pcMatRed) {
            _pcMatRed = new SLMaterial("Red", SLCol4f::RED);
            _pcMatRed->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
            _pcMatRed->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));
        }

        //remove old points
        if (_mapMesh) {
            _mapPC->deleteMesh(_mapMesh);
        }

        //add new (current) points
        SLVVec3f points, normals;
        const auto& mpts = _map->mapPoints();
        for (const auto& mpt : mpts) {
            points.push_back(mpt->worldPosVec());
            normals.push_back(mpt->normalVec());
        }
        _mapMesh = new SLPoints(points, normals, "MapPoints", _pcMatRed);
        _mapPC->addMesh(_mapMesh);
        _mapPC->updateAABBRec();
    }

    //draw matched map points
    if (_drawMapPointsMatches) {

    }
    //draw key frames
    if (_drawKeyFrames) {

    }
}
//-----------------------------------------------------------------------------
bool SLCVTrackedMapping::Relocalization()
{
    // Compute Bag of Words Vector
    SLAverageTiming::start("ComputeBoW", 9, 2);
    mCurrentFrame.ComputeBoW();
    SLAverageTiming::stop("ComputeBoW");

    // Relocalization is performed when tracking is lost
    // Track Lost: Query SLCVKeyFrame Database for keyframe candidates for relocalisation
    SLAverageTiming::start("DetectRelocalizationCandidates", 10, 2);
    vector<SLCVKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);
    SLAverageTiming::stop("DetectRelocalizationCandidates");

    if (vpCandidateKFs.empty())
        return false;

    //vector<SLCVKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->keyFrames();
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver

    SLAverageTiming::start("MatchCandsAndSolvePose", 11, 2);
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
            cout << "Num matches: " << nmatches << endl;
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

    SLAverageTiming::start("SearchCandsUntil50Matches", 12, 2);
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
    SLAverageTiming::start("UpdateLocalMap", 21, 2);
    UpdateLocalMap();
    SLAverageTiming::stop("UpdateLocalMap");

    SLAverageTiming::start("SearchLocalPoints", 22, 2);
    SearchLocalPoints();
    SLAverageTiming::stop("SearchLocalPoints");

    // Optimize Pose
    SLAverageTiming::start("PoseOptimizationTLM", 23, 2);
    Optimizer::PoseOptimization(&mCurrentFrame);
    SLAverageTiming::stop("PoseOptimizationTLM");
    mnMatchesInliers = 0;

    SLAverageTiming::start("UpdateMapPointsStat", 24, 2);
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
    SLAverageTiming::stop("UpdateMapPointsStat");

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