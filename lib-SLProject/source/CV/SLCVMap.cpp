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

//-----------------------------------------------------------------------------
SLCVMap::SLCVMap(const string& name)
    : mnMaxKFid(0)
{

}
//-----------------------------------------------------------------------------
SLCVMap::~SLCVMap()
{
    //for (auto* pt : _mapPoints) {
    //    if (pt)
    //        delete pt;
    //}
    for (auto* pt : mspMapPoints) {
        if (pt)
            delete pt;
    }
}
//-----------------------------------------------------------------------------
//! get visual representation as SLPoints
SLPoints* SLCVMap::getSceneObject()
{
    if (!_sceneObject)
    {
        _sceneObject = getNewSceneObject();
    }
    else
    {
        //todo: check if something has changed (e.g. size) and manipulate object
    }

    return _sceneObject;
}
//-----------------------------------------------------------------------------
//! get visual representation as SLPoints
SLPoints* SLCVMap::getNewSceneObject()
{
    //make a new SLPoints object
    SLMaterial* pcMat1 = new SLMaterial("Red", SLCol4f::RED);
    pcMat1->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    pcMat1->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 2.0f));

    //get points as Vec3f and collect normals
    SLVVec3f points, normals;
    for (auto mapPt : mspMapPoints) {
        points.push_back(mapPt->worldPosVec());
        normals.push_back(mapPt->normalVec());
    }

    _sceneObject = new SLPoints(points, normals, "MapPoints", pcMat1);
    //vectos must habe the same size
    return _sceneObject;
}
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