//#############################################################################
//  File:      SLDisk.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLDisk.h>

#include <utility>

//-----------------------------------------------------------------------------
/*!
SLDisk::SLDisk ctor for disk revolution object around the z-axis
*/
SLDisk::SLDisk(SLAssetManager* assetMgr,
               SLfloat         radius,
               const SLVec3f&  revolveAxis,
               SLuint          slices,
               SLbool          doubleSided,
               SLstring        name,
               SLMaterial*     mat) : SLRevolver(assetMgr, std::move(name))
{
    assert(slices >= 3 && "Error: Not enough slices.");

    _radius      = radius;
    _doubleSided = doubleSided;
    _slices      = slices;
    _smoothFirst = true;
    _revAxis     = revolveAxis;

    // add the centre & radius point
    if (_doubleSided)
    {
        _revPoints.reserve(4);
        _revPoints.push_back(SLVec3f(0, 0, 0));
        _revPoints.push_back(SLVec3f(radius, 0, 0));
        _revPoints.push_back(SLVec3f(radius, 0, 0));
        _revPoints.push_back(SLVec3f(0, 0, 0));
        _smoothLast = true;
    }
    else
    {
        _revPoints.reserve(2);
        _revPoints.push_back(SLVec3f(0, 0, 0));
        _revPoints.push_back(SLVec3f(radius, 0, 0));
        _smoothLast = false;
    }

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
