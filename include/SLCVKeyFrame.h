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
    //setters
    void id(int id) { _id = id; }
    void wTc(const SLCVMat& wTc) { _wTc = wTc; }
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

    // Variables used by the keyframe database
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

private:
    int _id = -1;
    //! opencv coordinate representation: z-axis points to principlal point,
    //! x-axis to the right and y-axis down
    //! Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
    SLCVMat _wTc;

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

    //image feature descriptors
    SLCVMat _descriptors;
};

typedef std::vector<SLCVKeyFrame> SLCVVKeyFrame;

#endif // !SLCVKEYFRAME_H
