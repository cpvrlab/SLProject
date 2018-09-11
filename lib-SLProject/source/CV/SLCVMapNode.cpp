//#############################################################################
//  File:      SLCVMapNode.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVMap.h>
#include <SLCVMapNode.h>
#include <SLMaterial.h>
#include <SLPoints.h>
#include <SLApplication.h>
#include <SLCVCalibration.h>
#include <SLCVCamera.h>
#include <SLCVKeyFrame.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
SLCVMapNode::SLCVMapNode(std::string name)
    : SLNode(name),
    _keyFrames(new SLNode("KeyFrames")),
    _mapPC(new SLNode("MapPC")),
    _mapMatchedPC(new SLNode("MapMatchedPC")),
    _mapLocalPC(new SLNode("MapLocalPC"))
{
    init();
}
//-----------------------------------------------------------------------------
SLCVMapNode::SLCVMapNode(std::string name, SLCVMap& map)
    : SLNode(name),
    _keyFrames(new SLNode("KeyFrames")),
    _mapPC(new SLNode("MapPC")),
    _mapMatchedPC(new SLNode("MapMatchedPC")),
    _mapLocalPC(new SLNode("MapLocalPC"))
{
    init();
    setMap(map);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::setMap(SLCVMap& map)
{
    map.setMapNode(this);
    updateAll(map);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::doUpdate()
{
    if (_removeMapPoints) {
        _mutex.lock();
        _removeMapPoints = false;
        _mutex.unlock();

        if (_mapMesh)
            _mapPC->deleteMesh(_mapMesh);
    }

    if (_removeMapPointsLocal) {
        _mutex.lock();
        _removeMapPointsLocal = false;
        _mutex.unlock();

        if (_mapLocalMesh)
            _mapLocalPC->deleteMesh(_mapLocalMesh);
    }

    if (_removeMapPointsMatched) {
        _mutex.lock();
        _removeMapPointsMatched = false;
        _mutex.unlock();

        if (_mapMatchesMesh)
            _mapMatchedPC->deleteMesh(_mapMatchesMesh);
    }

    if (_removeKeyFrames) {
        _mutex.lock();
        _removeKeyFrames = false;
        _mutex.unlock();

        if (_keyFrames)
            _keyFrames->deleteChildren();
    }

    if (_mapPtsChanged) {
        _mutex.lock();
        std::vector<SLCVMapPoint*> mapPts = _mapPts;
        _mapPtsChanged = false;
        _mutex.unlock();

        doUpdateMapPoints("MapPoints", mapPts, _mapPC, _mapMesh, _pcMat);
    }

    if (_mapPtsLocalChanged) {
        _mutex.lock();
        std::vector<SLCVMapPoint*> mapPtsLocal = _mapPtsLocal;
        _mapPtsLocalChanged = false;
        _mutex.unlock();

        doUpdateMapPoints("MapPointsLocal", mapPtsLocal, _mapLocalPC, _mapLocalMesh, _pcLocalMat);
    }

    if (_mapPtsMatchedChanged) {
        _mutex.lock();
        std::vector<SLCVMapPoint*> mapPtsMatched = _mapPtsMatched;
        _mapPtsMatchedChanged = false;
        _mutex.unlock();

        doUpdateMapPoints("MapPointsMatches", mapPtsMatched, _mapMatchedPC, _mapMatchesMesh, _pcMatchedMat);
    }

    if (_keyFramesChanged) {
        _mutex.lock();
        std::vector<SLCVKeyFrame*> kfs = _kfs;
        _keyFramesChanged = false;
        _mutex.unlock();

        doUpdateKeyFrames(kfs);
    }
}
//-----------------------------------------------------------------------------
void SLCVMapNode::init()
{
    //add map nodes for keyframes, mappoints, matched mappoints and local mappoints
    addChild(_keyFrames);
    addChild(_mapPC);
    addChild(_mapMatchedPC);
    addChild(_mapLocalPC);

    //instantiate materials
    _pcMat = new SLMaterial("Red", SLCol4f::RED);
    _pcMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    _pcMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 2.0f));

    _pcMatchedMat = new SLMaterial("Green", SLCol4f::GREEN);
    _pcMatchedMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    _pcMatchedMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    _pcLocalMat = new SLMaterial("Magenta", SLCol4f::MAGENTA);
    _pcLocalMat->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
    _pcLocalMat->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 4.0f));
}
//-----------------------------------------------------------------------------
void SLCVMapNode::clearAll()
{
    lock_guard<mutex> guard(_mutex);
    //remove and delete all old meshes, if existant
    _removeMapPoints = true;
    _removeMapPointsLocal = true;
    _removeMapPointsMatched = true;
    _removeKeyFrames = true;
    _mapPtsChanged = false;
    _mapPtsLocalChanged = false;
    _mapPtsMatchedChanged = false;
    _keyFramesChanged = false;
    _mapPts.clear();
    _mapPtsLocal.clear();
    _mapPtsMatched.clear();
    _kfs.clear();
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateAll(SLCVMap& map) //todo: const SLCVMap
{
    //remove and delete all old meshes, if existant
    removeMapPointsLocal();
    removeMapPointsMatched();
    //remove and reinsert map points and keyframes
    updateMapPoints(map.GetAllMapPoints());
    updateKeyFrames( map.GetAllKeyFrames());
}
//-----------------------------------------------------------------------------
void SLCVMapNode::doUpdateMapPoints(std::string name, const std::vector<SLCVMapPoint*>& pts,
    SLNode*& node, SLPoints*& mesh, SLMaterial*& material)
{
    //remove old mesh, if it exists
    if (mesh)
        node->deleteMesh(mesh);

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        SLVVec3f points, normals;
        for (auto mapPt : pts) {
            points.push_back(mapPt->worldPosVec());
            normals.push_back(mapPt->normalVec());
        }

        mesh = new SLPoints(points, normals, name, material);
        node->addMesh(mesh);
        node->updateAABBRec();
    }
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateMapPoints(const std::vector<SLCVMapPoint*>& pts)
{
    lock_guard<mutex> guard(_mutex);
    _mapPts = pts;
    _mapPtsChanged = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateMapPointsLocal(const std::vector<SLCVMapPoint*>& pts)
{
    lock_guard<mutex> guard(_mutex);
    _mapPtsLocal = pts;
    _mapPtsLocalChanged = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateMapPointsMatched(const std::vector<SLCVMapPoint*>& pts)
{
    lock_guard<mutex> guard(_mutex);
    _mapPtsMatched = pts;
    _mapPtsMatchedChanged = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateKeyFrames(const std::vector<SLCVKeyFrame*>& kfs)
{
    lock_guard<mutex> guard(_mutex);
    _keyFramesChanged = true;
    _kfs = kfs;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::doUpdateKeyFrames(const std::vector<SLCVKeyFrame*>& kfs)
{
    _keyFrames->deleteChildren();
    //Delete keyframe textures
    for(const auto& texture : _kfTextures)
    {
        SLApplication::scene->deleteTexture(texture);
    }
    _kfTextures.clear();

    for (auto* kf : kfs) {

        SLCVCamera* cam = new SLCVCamera(this, "KeyFrame" + kf->mnId);
        //set background
        if (kf->getTexturePath().size())
        {
            // TODO(jan): textures are saved in a global textures vector (scene->textures)
            // and should be deleted from there. Otherwise we have a yuuuuge memory leak.
#if 0
            SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
            _kfTextures.push_back(texture);
            cam->background().texture(texture);
#endif
        }

        cam->om(kf->getObjectMatrix());

        //calculate vertical field of view
        SLfloat fy = (SLfloat)kf->fy;
        SLfloat cy = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * SL_RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11);
        cam->clipNear(0.1);
        cam->clipFar(1000.0);
        _keyFrames->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeMapPoints()
{
    lock_guard<mutex> guard(_mutex);
    _removeMapPoints = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeMapPointsLocal()
{
    lock_guard<mutex> guard(_mutex);
    _removeMapPointsLocal = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeMapPointsMatched()
{
    lock_guard<mutex> guard(_mutex);
    _removeMapPointsMatched = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeKeyFrames()
{
    lock_guard<mutex> guard(_mutex);
    _removeKeyFrames = true;
}
//-----------------------------------------------------------------------------
void SLCVMapNode::setHideMapPoints(bool state)
{
    if(_mapPC->drawBits()->get(SL_DB_HIDDEN) != state)
        _mapPC->drawBits()->set(SL_DB_HIDDEN, state);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::setHideKeyFrames(bool state)
{
    if (_keyFrames->drawBits()->get(SL_DB_HIDDEN) != state)
    {
        _keyFrames->drawBits()->set(SL_DB_HIDDEN, state);
        for (SLNode* child : _keyFrames->children()) {
            if (child)
                child->drawBits()->set(SL_DB_HIDDEN, state);
        }
    }
}
