//#############################################################################
//  File:      SLCVMap.cpp
//  Author:    Michael G�ttlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "stdafx.h"
#include "SLCVMap.h"
#include <SLMaterial.h>
#include <SLGLGenericProgram.h>
#include <SLPoints.h>
#include <SLCVKeyFrameDB.h>
#include <SLCVKeyFrame.h>
#include <SLCVMapNode.h>

using namespace cv;

//-----------------------------------------------------------------------------
SLCVMap::SLCVMap(const string& name)
    : mnMaxKFid(0),
    _mapNode(NULL)
{

}
//-----------------------------------------------------------------------------
SLCVMap::~SLCVMap()
{
    for (auto* pt : mspMapPoints) {
        if (pt)
            delete pt;
    }
}
//-----------------------------------------------------------------------------
////! get visual representation as SLPoints
//SLPoints* SLCVMap::getSceneObject()
//{
//    if (!_sceneObject)
//    {
//        _sceneObject = getNewSceneObject();
//    }
//    else
//    {
//        //todo: check if something has changed (e.g. size) and manipulate object
//    }
//
//    return _sceneObject;
//}
//-----------------------------------------------------------------------------
////! get visual representation as SLPoints
//SLPoints* SLCVMap::getNewSceneObject()
//{
//    //make a new SLPoints object
//    SLMaterial* pcMat1 = new SLMaterial("Red", SLCol4f::RED);
//    pcMat1->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
//    pcMat1->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 2.0f));
//
//    //get points as Vec3f and collect normals
//    SLVVec3f points, normals;
//    for (auto mapPt : mspMapPoints) {
//        points.push_back(mapPt->worldPosVec());
//        normals.push_back(mapPt->normalVec());
//    }
//
//    _sceneObject = new SLPoints(points, normals, "MapPoints", pcMat1);
//    //vectos must habe the same size
//    return _sceneObject;
//}
//-----------------------------------------------------------------------------
void SLCVMap::clear()
{
    for (auto* pt : mspMapPoints) {
        if (pt)
            delete pt;
    }
    for (auto* kf : mspKeyFrames) {
        if (kf)
            delete kf;
    }
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();

    //remove visual representation
    if (_mapNode)
        _mapNode->updateAll(*this);
}
//-----------------------------------------------------------------------------
void SLCVMap::SetReferenceMapPoints(const vector<SLCVMapPoint*> &vpMPs)
{
    //unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}
//-----------------------------------------------------------------------------
long unsigned int SLCVMap::KeyFramesInMap()
{
    //unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
    //return mpKeyFrameDatabase->keyFrames().size();
}
//-----------------------------------------------------------------------------
void SLCVMap::EraseMapPoint(SLCVMapPoint *pMP)
{
    //unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}
//-----------------------------------------------------------------------------
vector<SLCVMapPoint*> SLCVMap::GetAllMapPoints()
{
    //unique_lock<mutex> lock(mMutexMap);
    return vector<SLCVMapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}
//-----------------------------------------------------------------------------
const std::set<SLCVMapPoint*>& SLCVMap::GetAllMapPointsConstRef() const
{
    //unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints;
}
//-----------------------------------------------------------------------------
std::set<SLCVMapPoint*>& SLCVMap::GetAllMapPointsRef()
{
    //unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints;
}
//-----------------------------------------------------------------------------
void SLCVMap::AddMapPoint(SLCVMapPoint *pMP)
{
    //unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}
//-----------------------------------------------------------------------------
void SLCVMap::AddKeyFrame(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if (pKF->id()>mnMaxKFid)
        mnMaxKFid = pKF->id();
}
//-----------------------------------------------------------------------------
long unsigned int SLCVMap::MapPointsInMap()
{
    //unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}
//-----------------------------------------------------------------------------
void SLCVMap::EraseKeyFrame(SLCVKeyFrame *pKF)
{
    //unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}
////-----------------------------------------------------------------------------
//SLCVMapPoint* SLCVMap::getMapPointForId(int id)
//{
//    if (mspMapPoints.find(id) != mspMapPoints.end())
//        return mspMapPoints[id];
//    else
//        return NULL;
//}
//-----------------------------------------------------------------------------
std::vector<SLCVKeyFrame*> SLCVMap::GetAllKeyFrames()
{
    return vector<SLCVKeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}
//-----------------------------------------------------------------------------
void SLCVMap::rotate(float value, int type)
{
    //transform to degree
    value *= SL_DEG2RAD;

    Mat rot = buildRotMat(value, type);
    cout << "rot: " << rot << endl;

    //rotate keyframes
    Mat Twc;
    for (auto& kf : mspKeyFrames)
    {
        //get and rotate
        Twc = kf->GetPose().inv();
        Twc = rot * Twc;
        //set back
        kf->SetPose(Twc.inv());
    }

    //rotate keypoints
    Mat Pw;
    Mat rot33 = rot.rowRange(0, 3).colRange(0, 3);
    for (auto& pt : mspMapPoints)
    {
        Pw = rot33 * pt->worldPos();
        pt->worldPos(rot33 * pt->worldPos());
    }
}
//-----------------------------------------------------------------------------
void SLCVMap::translate(float value, int type)
{
    Mat trans = buildTransMat(value, type);

    cout << "trans: " << trans << endl;

    //rotate keyframes
    Mat Twc;
    for (auto& kf : mspKeyFrames)
    {
        //get and translate
        cv::Mat Twc = kf->GetPose().inv();
        Twc.rowRange(0, 3).col(3) += trans;
        //set back
        kf->SetPose(Twc.inv());
    }

    //rotate keypoints
    for (auto& pt : mspMapPoints)
    {
        pt->worldPos(trans + pt->worldPos());
    }
}
//-----------------------------------------------------------------------------
void SLCVMap::scale(float value)
{
    for (auto& kf : mspKeyFrames)
    {
        //get and translate
        cv::Mat Tcw = kf->GetPose();
        Tcw.rowRange(0, 3).col(3) *= value;
        //set back
        kf->SetPose(Tcw);
    }

    //rotate keypoints
    for (auto& pt : mspMapPoints)
    {
        pt->worldPos(value * pt->worldPos());
    }
}
//-----------------------------------------------------------------------------
void SLCVMap::applyTransformation(double value, TransformType type)
{
    //apply rotation, translation and scale to Keyframe and MapPoint poses
    cout << "apply transform with value: " << value << endl;
    switch (type)
    {
    case ROT_X:
        //build different transformation matrices for x,y and z rotation
        rotate((float)value, 0);
        break;
    case ROT_Y:
        rotate((float)value, 1);
        break;
    case ROT_Z:
        rotate((float)value, 2);
        break;
    case TRANS_X:
        translate((float)value, 0);
        break;
    case TRANS_Y:
        translate((float)value, 1);
        break;
    case TRANS_Z:
        translate((float)value, 2);
        break;
    case SCALE:
        scale((float)value);
        break;
    }

    //compute resulting values for map points
    for (auto& mp : mspMapPoints)
    {
        //mean viewing direction and depth
        mp->UpdateNormalAndDepth();
        mp->ComputeDistinctiveDescriptors();
    }

    //update scene objects
    //exchange all Keyframes (also change name)
    if (_mapNode)
    {
        _mapNode->updateAll(*this);
    }
}
//-----------------------------------------------------------------------------
// Build rotation matrix
Mat SLCVMap::buildTransMat(float &val, int type)
{
    Mat trans = cv::Mat::zeros(3, 1, CV_32F);
    switch (type)
    {
    case 0:
        trans.at<float>(0, 0) = val;
        break;

    case 1:
        //!!turn sign of y coordinate
        trans.at<float>(1, 0) = -val;
        break;

    case 2:
        //!!turn sign of z coordinate
        trans.at<float>(2, 0) = -val;
        break;
    }

    return trans;
}
//-----------------------------------------------------------------------------
// Build rotation matrix
Mat SLCVMap::buildRotMat(float &valDeg, int type)
{
    Mat rot = Mat::ones(4, 4, CV_32F);

    switch (type)
    {
    case 0:
        // Calculate rotation about x axis
        rot = (Mat_<float>(4, 4) <<
            1, 0, 0, 0,
            0, cos(valDeg), -sin(valDeg), 0,
            0, sin(valDeg), cos(valDeg), 0,
            0, 0, 0, 1
            );
        break;

    case 1:
        // Calculate rotation about y axis
        rot = (Mat_<float>(4, 4) <<
            cos(valDeg), 0, sin(valDeg), 0,
            0, 1, 0, 0,
            -sin(valDeg), 0, cos(valDeg), 0,
            0, 0, 0, 1
            );
        //invert direction for Y
        rot = rot.inv();
        break;

    case 2:
        // Calculate rotation about z axis
        rot = (Mat_<float>(4, 4) <<
            cos(valDeg), -sin(valDeg), 0, 0,
            sin(valDeg), cos(valDeg), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
            );
        //invert direction for Z
        rot = rot.inv();
        break;
    }

    return rot;
}
//-----------------------------------------------------------------------------
void SLCVMap::setMapNode(SLCVMapNode* mapNode)
{
    _mapNode = mapNode;
}