//#############################################################################
//  File:      SLSpheric.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSpheric.h>

#include <utility>

//-----------------------------------------------------------------------------
/*!
SLSpheric::SLSpheric ctor for spheric revolution object around the z-axis.
*/
SLSpheric::SLSpheric(SLAssetManager* assetMgr,
                     SLfloat         sphereRadius,
                     SLfloat         thetaStartDEG,
                     SLfloat         thetaEndDEG,
                     SLuint          stacks,
                     SLuint          slices,
                     SLstring        name,
                     SLMaterial*     mat) : SLRevolver(assetMgr, std::move(name))
{
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices > 0 && "Error: Not enough stacks.");
    assert(thetaStartDEG >= 0.0f && thetaStartDEG < 180.0f &&
           "Error: Polar start angle < 0 or > 180");
    assert(thetaEndDEG > 0.0f && thetaEndDEG <= 180.0f &&
           "Error: Polar end angle < 0 or > 180");
    assert(thetaStartDEG < thetaEndDEG &&
           "Error: Polar start angle > end angle");

    _radius        = sphereRadius;
    _thetaStartDEG = thetaStartDEG;
    _thetaEndDEG   = thetaEndDEG;

    _stacks      = stacks;
    _slices      = slices;
    _smoothFirst = (thetaStartDEG == 0.0f);
    _smoothLast  = (thetaEndDEG == 180.0f);
    _isVolume    = (thetaStartDEG == 0.0f && thetaEndDEG == 180.0f);
    _revAxis.set(0, 0, 1);
    _revPoints.reserve(stacks + 1);

    SLfloat theta  = -Utils::PI + thetaStartDEG * Utils::DEG2RAD;
    SLfloat phi    = 0;
    SLfloat dTheta = (thetaEndDEG - thetaStartDEG) * Utils::DEG2RAD / stacks;

    for (SLuint i = 0; i <= stacks; ++i)
    {
        SLVec3f p;
        p.fromSpherical(sphereRadius, theta, phi);
        _revPoints.push_back(p);
        theta += dTheta;
    }

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
