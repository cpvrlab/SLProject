//#############################################################################
//  File:      SLCVTrackedRaulMur.cpp
//  Author:    Michael Göttlicher
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
#include <OrbSlam\ORBmatcher.h>
#include <OrbSlam\PnPsolver.h>

using namespace cv;
using namespace ORB_SLAM2;

//-----------------------------------------------------------------------------
SLCVTrackedRaulMur::SLCVTrackedRaulMur(SLNode *node, ORBVocabulary* vocabulary, 
    SLCVKeyFrameDB* keyFrameDB)
    : SLCVTracked(node),
    mpVocabulary(vocabulary),
    mpKeyFrameDatabase(keyFrameDB)
{
    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    //instantiate Orb extractor
    _extractor = new ORBextractor(1500, 1.44f, 4, 30, 20);
}
//-----------------------------------------------------------------------------
SLCVTrackedRaulMur::~SLCVTrackedRaulMur()
{
    if (_extractor)
        delete _extractor;
}
//-----------------------------------------------------------------------------
SLbool SLCVTrackedRaulMur::track(SLCVMat imageGray,
    SLCVMat image,
    SLCVCalibration* calib,
    SLbool drawDetection,
    SLSceneView* sv)
{
    /************************************************************/
    //Frame constructor call in ORB-SLAM:
    // Current Frame
    double timestamp = 0.0; //todo
    mCurrentFrame = SLCVFrame(imageGray, timestamp, _extractor, 
        calib->cameraMat(), calib->distortion(), mpVocabulary );
    /************************************************************/
    //Track():
    //if (no last pos)
    Relocalization();

    //else
    // TrackWithMotionModel()
    //{
    //  if no valid result -> Relocalization()
    //}

    //if enough matches, then track local map

    //update motion model

    //clean up

    //store last frame
    /************************************************************/

    //set new camera position
    
    return false;
}
//-----------------------------------------------------------------------------
bool SLCVTrackedRaulMur::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<SLCVKeyFrame*> vpCandidateKFs = mpKeyFrameDatabase->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

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

                // If few inliers, search by projection in a coarse window and optimize again
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