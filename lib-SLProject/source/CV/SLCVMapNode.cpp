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
    //set this map not in the map for updates of scene objects after map transformations
    map.setMapNode(this);

    init();
    updateAll(map);
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
    doUpdateMapPoints("MapPoints", pts, _mapPC, _mapMesh, _pcMat);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateMapPointsLocal(const std::vector<SLCVMapPoint*>& pts)
{
    doUpdateMapPoints("MapPointsLocal", pts, _mapLocalPC, _mapLocalMesh, _pcLocalMat);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateMapPointsMatched(const std::vector<SLCVMapPoint*>& pts)
{
    doUpdateMapPoints("MapPointsMatches", pts, _mapMatchedPC, _mapMatchesMesh, _pcMatchedMat);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::updateKeyFrames(const std::vector<SLCVKeyFrame*>& kfs)
{
    _keyFrames->deleteChildren();
    for (auto* kf : kfs) {

        SLCVCamera* cam = new SLCVCamera(this, "KeyFrame" + kf->mnId);
        //set background
        if (kf->getTexturePath().size())
        {
            SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
            cam->background().texture(texture);
        }

        cam->om(kf->getObjectMatrix());

        cam->fov(SLApplication::activeCalib->cameraFovDeg());
        cam->focalDist(0.11);
        cam->clipNear(0.1);
        cam->clipFar(1000.0);
        _keyFrames->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeMapPoints()
{
    if (_mapMesh)
        _mapPC->deleteMesh(_mapMesh);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeMapPointsLocal()
{
    if (_mapLocalMesh)
        _mapLocalPC->deleteMesh(_mapLocalMesh);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeMapPointsMatched()
{
    if (_mapMatchesMesh)
        _mapMatchedPC->deleteMesh(_mapMatchesMesh);
}
//-----------------------------------------------------------------------------
void SLCVMapNode::removeKeyFrames()
{
    _keyFrames->deleteChildren();
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