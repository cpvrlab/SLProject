//#############################################################################
//  File:      SLCVKeyframe.h
//  Author:    Michael Göttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVKEYFRAME_H
#define SLCVKEYFRAME_H

#include <vector>
#include <SLCamera.h>
//#include <SLCVMapPoint.h>
#include <DBoW2/DBoW2/BowVector.h>
#include <DBoW2/DBoW2/FeatureVector.h>
#include <OrbSlam\ORBVocabulary.h>

using namespace ORB_SLAM2;

class SLCVMapPoint;
//-----------------------------------------------------------------------------
//! AR Keyframe node class
/*! A Keyframe is a camera with a position and additional information about key-
points that were found in this frame. It also contains descriptors for the found
keypoints.
*/
class SLCVKeyFrame
{
public:
    SLCVKeyFrame();

    //getters
    int id() { return _id; }
    vector<SLCVMapPoint*> GetMapPointMatches() {
        //unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints; }
    const SLCVMat& descriptors() { return _descriptors; }
    cv::Mat GetCameraCenter();

    //setters
    void id(int id) { _id = id; }

    void Tcw(const SLCVMat& Tcw) {
        //_wTc = wTc; 
        //unique_lock<mutex> lock(mMutexPose);
        Tcw.copyTo(_Tcw);
        cv::Mat Rcw = _Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = _Tcw.rowRange(0, 3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc*tcw;

        _Twc = cv::Mat::eye(4, 4, Tcw.type());
        Rwc.copyTo(_Twc.rowRange(0, 3).colRange(0, 3));
        Ow.copyTo(_Twc.rowRange(0, 3).col(3));
        //cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
        //Cw = Twc*center;
    }
    void descriptors(const SLCVMat& descriptors) { _descriptors = descriptors; }
    //! get visual representation as SLPoints
    SLCamera* getSceneObject();

    // Covisibility graph functions
    vector<SLCVKeyFrame*> GetBestCovisibilityKeyFrames(const int &N);

    // MapPoint observation functions
    void AddMapPoint(SLCVMapPoint* pMP, size_t idx);

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Bag of Words Representation
    void ComputeBoW(ORBVocabulary* orbVocabulary);

    // Covisibility graph functions
    void AddConnection(SLCVKeyFrame* pKF, int weight);
    void UpdateBestCovisibles();
    void UpdateConnections();

    // Spanning tree functions
    void AddChild(SLCVKeyFrame* pKF);

    bool isBad() { return false; }

    // Variables used by the keyframe database
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    //undistorted keypoints
    std::vector<cv::KeyPoint> mvKeysUn;

    // Scale
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;

private:
    int _id = -1;
    //! opencv coordinate representation: z-axis points to principlal point,
    //! x-axis to the right and y-axis down
    //! Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
    SLCVMat _Twc;
    SLCVMat _Tcw;
    //! camera center
    SLCVMat Ow;

    // MapPoints associated to keypoints (this array contains NULL for every
    //unassociated keypoint from original frame)
    std::vector<SLCVMapPoint*> mvpMapPoints;

    std::map<SLCVKeyFrame*, int> mConnectedKeyFrameWeights;
    std::vector<SLCVKeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection = true;
    SLCVKeyFrame* mpParent = NULL;
    std::set<SLCVKeyFrame*> mspChildrens;
    std::set<SLCVKeyFrame*> mspLoopEdges;

    //Pointer to visual representation object (ATTENTION: do not delete this object)
    //We do not use inheritence, because the scene is responsible for all scene objects!
    SLCamera* _camera = NULL;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)

    //image feature descriptors
    SLCVMat _descriptors;
};

typedef std::vector<SLCVKeyFrame> SLCVVKeyFrame;

#endif // !SLCVKEYFRAME_H
