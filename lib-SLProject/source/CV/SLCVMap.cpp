//#############################################################################
//  File:      SLCVMap.cpp
//  Author:    Michael Göttlicher
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

//-----------------------------------------------------------------------------
SLCVMap::SLCVMap(const string& name="Map")
{

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
    for (auto mapPt : _mapPoints) {
        points.push_back(mapPt.worldPosVec());
        normals.push_back(mapPt.normalVec());
    }

    _sceneObject = new SLPoints(points, normals, "MapPoints", pcMat1);
    //vectos must habe the same size
    return _sceneObject;
}