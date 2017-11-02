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
        //make a new SLPoints object
        SLMaterial* pcMat1 = new SLMaterial("Red", SLCol4f::RED);
        pcMat1->program(new SLGLGenericProgram("ColorUniformPoint.vert", "Color.frag"));
        pcMat1->program()->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

        //get points as Vec3f
        SLVVec3f points;
        for (auto mapPt : _mapPoints)
            points.push_back(mapPt.vec3f());

        _sceneObject = new SLPoints(points, "MapPoints", pcMat1);
    }
    else
    {
        //todo: check if something has changed (e.g. size) and manipulate object
    }

    return _sceneObject;
}
//-----------------------------------------------------------------------------