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

//-----------------------------------------------------------------------------
SLCVMapNode::SLCVMapNode(SLCVMap* map, std::string name)
    : _keyFrames(new SLNode("KeyFrames")),
    _mapPC(new SLNode("MapPC")),
    _mapMatchedPC(new SLNode("MapMatchedPC")),
    _mapLocalPC(new SLNode("MapLocalPC"))
{
    //add map nodes for keyframes, mappoints, matched mappoints and local mappoints
    addChild(_keyFrames);
    addChild(_mapPC);
    addChild(_mapMatchedPC);

    //instantiate and assign meshes and materials to 
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
SLCVMapNode::~SLCVMapNode()
{
    //todo: wann müssen wir deleten? nur wenn nicht der scene angefügt??
    if (_keyFrames)
        delete _keyFrames;
    if (_mapPC)
        delete _mapPC;
    if (_mapMatchedPC)
        delete _mapMatchedPC;
    if (_mapLocalPC)
        delete _mapLocalPC;

    //todo: ggf material und meshes deleten??
}
//-----------------------------------------------------------------------------
void SLCVMapNode::addMapObjects(SLCVMap& map)
{
    //map points
    //todo: komplett hier!!!
    SLPoints* mapPtsMesh = map.getSceneObject();
    _mapPC->addMesh(mapPtsMesh);

    ////add additional empty point clouds for visualization of local map and map point matches:
    ////map point matches
    ////todo: notwendig einen punkt einzufügen???
    //SLVVec3f points, normals;
    //points.push_back(SLVec3f(0.f, 0.f, 0.f));
    //normals.push_back(SLVec3f(0.0001f, 0.0001f, 0.0001f));
    //SLPoints* mapMatchesMesh = new SLPoints(points, normals, "MapPointsMatches", _pcMatchedMat);
    //_mapLocalPC->addMesh(mapMatchesMesh);

    ////local map points mesh
    //SLPoints* mapLocalMesh = new SLPoints(points, normals, "MapPointsLocal", _pcLocalMat);
    //_mapLocalPC->addMesh(mapLocalMesh);

    //add keyFrames
    auto keyFrames = map.GetAllKeyFrames();
    for (auto* kf : keyFrames) {
        SLCVCamera* cam = kf->getSceneObject();
        cam->fov(SLApplication::activeCalib->cameraFovDeg());
        cam->focalDist(0.11);
        cam->clipNear(0.1);
        cam->clipFar(1000.0);
        _keyFrames->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void SLCVMapNode::doUpdateMapPoints(std::string name, const std::vector<SLCVMapPoint*>& pts,
    SLNode* node, SLMesh* mesh, SLMaterial* material)
{
    //remove old mesh, if it exists
    if (_mapMatchesMesh)
    {
        _mapMatchedPC->deleteMesh(_mapMatchesMesh);
    }

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